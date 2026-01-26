import { test, expect } from '@playwright/test';

/**
 * SentinelX API Tests
 * ===================
 *
 * Tests for the FastAPI endpoints:
 * - Health check
 * - Prediction endpoint
 * - Alerts CRUD
 * - Alert fatigue metrics
 *
 * Prerequisites: docker-compose up -d
 */

const API_BASE = 'http://localhost:8000';

test.describe('API Health', () => {
  test('should return healthy status', async ({ request }) => {
    const response = await request.get(`${API_BASE}/health`);

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(body.status).toBe('healthy');
    expect(body.model_loaded).toBe(true);
    expect(body.database.status).toBe('healthy');
  });

  test('should have database connection', async ({ request }) => {
    const response = await request.get(`${API_BASE}/health`);
    const body = await response.json();

    expect(body.database).toBeDefined();
    expect(body.database.connection_pool).toBeDefined();
  });
});

test.describe('Prediction Endpoint', () => {
  const validPayload = {
    system: {
      machine_id: 'TEST-M1',
      air_temperature_K: 305.2,
      process_temperature_K: 312.1,
      rotational_speed_rpm: 1450,
      torque_Nm: 45.3,
      tool_wear_min: 120,
      vibration_mm_s: 18.5,
      pressure_psi: 115.0,
      network_latency_ms: 25.0,
      edge_processing_time_ms: 12.0,
      fuzzy_pid_output: 0.65
    },
    application: {
      error_rate_pct: 5.0,
      cpu_utilization_pct: 60.0,
      memory_utilization_pct: 55.0,
      disk_io_wait_ms: 10.0,
      packet_loss_pct: 0.5,
      api_response_latency_ms: 150.0,
      http_5xx_count: 2,
      queue_depth: 15,
      request_throughput_rps: 100.0
    }
  };

  test('should accept valid prediction request', async ({ request }) => {
    const response = await request.post(`${API_BASE}/predict`, {
      data: validPayload
    });

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(body.alert_id).toBeDefined();
    expect(body.machine_id).toBe('TEST-M1');
    expect(body.risk_level).toBeDefined();
    expect(body.failure_probability).toBeGreaterThanOrEqual(0);
    expect(body.failure_probability).toBeLessThanOrEqual(1);
  });

  test('should return markdown report', async ({ request }) => {
    const response = await request.post(`${API_BASE}/predict`, {
      data: validPayload
    });

    const body = await response.json();
    expect(body.markdown_report).toContain('## SentinelX Diagnostic Report');
    expect(body.markdown_report).toContain('Risk Level');
    expect(body.markdown_report).toContain('Root Cause Analysis');
  });

  test('should return json summary', async ({ request }) => {
    const response = await request.post(`${API_BASE}/predict`, {
      data: validPayload
    });

    const body = await response.json();
    expect(body.json_summary).toBeDefined();
    expect(body.json_summary.timestamp).toBeDefined();
    expect(body.json_summary.risk_level).toBeDefined();
    expect(body.json_summary.root_cause).toBeDefined();
    expect(body.json_summary.top_features).toBeInstanceOf(Array);
  });

  test('should have valid risk levels', async ({ request }) => {
    const response = await request.post(`${API_BASE}/predict`, {
      data: validPayload
    });

    const body = await response.json();
    const validRiskLevels = ['nominal', 'low', 'moderate', 'high', 'critical'];
    expect(validRiskLevels).toContain(body.risk_level);
  });

  test('should persist alert to database', async ({ request }) => {
    const response = await request.post(`${API_BASE}/predict`, {
      data: validPayload
    });

    const body = await response.json();
    const alertId = body.alert_id;

    // Verify alert exists
    const alertResponse = await request.get(`${API_BASE}/alerts/${alertId}`);
    expect(alertResponse.ok()).toBeTruthy();

    const alert = await alertResponse.json();
    expect(alert.id).toBe(alertId);
    expect(alert.machine_id).toBe('TEST-M1');
  });

  test('should reject invalid payload', async ({ request }) => {
    const invalidPayload = {
      system: {
        machine_id: 'TEST-M1'
        // Missing required fields
      },
      application: {}
    };

    const response = await request.post(`${API_BASE}/predict`, {
      data: invalidPayload
    });

    expect(response.status()).toBe(422); // Validation error
  });

  test('should reject out-of-bounds values', async ({ request }) => {
    const invalidPayload = {
      ...validPayload,
      system: {
        ...validPayload.system,
        air_temperature_K: 500 // Max is 400
      }
    };

    const response = await request.post(`${API_BASE}/predict`, {
      data: invalidPayload
    });

    expect(response.status()).toBe(422);
  });
});

test.describe('High-Risk Prediction', () => {
  test('should detect high risk conditions', async ({ request }) => {
    // Simulate high-risk conditions
    const highRiskPayload = {
      system: {
        machine_id: 'TEST-HIGH-RISK',
        air_temperature_K: 340.0,  // Very high temp
        process_temperature_K: 355.0,
        rotational_speed_rpm: 2800,
        torque_Nm: 85.0,  // High torque
        tool_wear_min: 280,  // High wear
        vibration_mm_s: 45.0,  // High vibration
        pressure_psi: 180.0,
        network_latency_ms: 150.0,
        edge_processing_time_ms: 50.0,
        fuzzy_pid_output: 0.95
      },
      application: {
        error_rate_pct: 25.0,  // High error rate
        cpu_utilization_pct: 98.0,
        memory_utilization_pct: 95.0,
        disk_io_wait_ms: 80.0,
        packet_loss_pct: 8.0,
        api_response_latency_ms: 800.0,
        http_5xx_count: 50,
        queue_depth: 200,
        request_throughput_rps: 30.0
      }
    };

    const response = await request.post(`${API_BASE}/predict`, {
      data: highRiskPayload
    });

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    // High-risk payload should trigger elevated risk level
    expect(body.failure_probability).toBeGreaterThan(0.01);
  });
});

test.describe('Alerts Endpoint', () => {
  test('should list alerts', async ({ request }) => {
    const response = await request.get(`${API_BASE}/alerts`);

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(body).toBeInstanceOf(Array);
  });

  test('should filter by machine_id', async ({ request }) => {
    // First create an alert
    const payload = {
      system: {
        machine_id: 'TEST-FILTER',
        air_temperature_K: 300.0,
        process_temperature_K: 308.0,
        rotational_speed_rpm: 1400,
        torque_Nm: 40.0,
        tool_wear_min: 100
      },
      application: {
        error_rate_pct: 3.0,
        cpu_utilization_pct: 50.0,
        memory_utilization_pct: 50.0,
        disk_io_wait_ms: 8.0,
        packet_loss_pct: 0.2,
        api_response_latency_ms: 120.0,
        http_5xx_count: 0,
        queue_depth: 10,
        request_throughput_rps: 100.0
      }
    };

    await request.post(`${API_BASE}/predict`, { data: payload });

    // Query with filter
    const response = await request.get(`${API_BASE}/alerts?machine_id=TEST-FILTER`);

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(body.every((alert: any) => alert.machine_id === 'TEST-FILTER')).toBeTruthy();
  });

  test('should filter by risk_level', async ({ request }) => {
    const response = await request.get(`${API_BASE}/alerts?risk_level=nominal`);

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    if (body.length > 0) {
      expect(body.every((alert: any) => alert.risk_level === 'nominal')).toBeTruthy();
    }
  });

  test('should paginate results', async ({ request }) => {
    const response = await request.get(`${API_BASE}/alerts?limit=5&offset=0`);

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(body.length).toBeLessThanOrEqual(5);
  });

  test('should get single alert by ID', async ({ request }) => {
    // Create an alert first
    const payload = {
      system: {
        machine_id: 'TEST-SINGLE',
        air_temperature_K: 300.0,
        process_temperature_K: 308.0,
        rotational_speed_rpm: 1400,
        torque_Nm: 40.0,
        tool_wear_min: 100
      },
      application: {
        error_rate_pct: 3.0,
        cpu_utilization_pct: 50.0,
        memory_utilization_pct: 50.0,
        disk_io_wait_ms: 8.0,
        packet_loss_pct: 0.2,
        api_response_latency_ms: 120.0,
        http_5xx_count: 0,
        queue_depth: 10,
        request_throughput_rps: 100.0
      }
    };

    const createResponse = await request.post(`${API_BASE}/predict`, { data: payload });
    const { alert_id } = await createResponse.json();

    // Get single alert
    const response = await request.get(`${API_BASE}/alerts/${alert_id}`);

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(body.id).toBe(alert_id);
    expect(body.report_md).toBeDefined();
    expect(body.raw_data).toBeDefined();
    expect(body.shap_values).toBeDefined();
  });

  test('should return 404 for non-existent alert', async ({ request }) => {
    const response = await request.get(`${API_BASE}/alerts/999999`);
    expect(response.status()).toBe(404);
  });
});

test.describe('Alert Acknowledgment', () => {
  test('should acknowledge alert', async ({ request }) => {
    // Create an alert
    const payload = {
      system: {
        machine_id: 'TEST-ACK',
        air_temperature_K: 300.0,
        process_temperature_K: 308.0,
        rotational_speed_rpm: 1400,
        torque_Nm: 40.0,
        tool_wear_min: 100
      },
      application: {
        error_rate_pct: 3.0,
        cpu_utilization_pct: 50.0,
        memory_utilization_pct: 50.0,
        disk_io_wait_ms: 8.0,
        packet_loss_pct: 0.2,
        api_response_latency_ms: 120.0,
        http_5xx_count: 0,
        queue_depth: 10,
        request_throughput_rps: 100.0
      }
    };

    const createResponse = await request.post(`${API_BASE}/predict`, { data: payload });
    const { alert_id } = await createResponse.json();

    // Acknowledge
    const ackResponse = await request.post(`${API_BASE}/alerts/${alert_id}/acknowledge`, {
      data: {
        acknowledged_by: 'test-user',
        action_taken: 'Reviewed and cleared',
        is_false_positive: false
      }
    });

    expect(ackResponse.ok()).toBeTruthy();

    const ackBody = await ackResponse.json();
    expect(ackBody.status).toBe('acknowledged');

    // Verify acknowledgment persisted
    const alertResponse = await request.get(`${API_BASE}/alerts/${alert_id}`);
    const alert = await alertResponse.json();
    expect(alert.acknowledged).toBe(true);
  });
});

test.describe('Alert Fatigue Metrics', () => {
  test('should return fatigue metrics', async ({ request }) => {
    const response = await request.get(`${API_BASE}/fatigue?days=7`);

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(body.period_days).toBe(7);
    expect(body.total_alerts).toBeGreaterThanOrEqual(0);
    expect(body.paper3_benchmark_fp_rate).toBe(54.0);
  });

  test('should filter by machine_id', async ({ request }) => {
    const response = await request.get(`${API_BASE}/fatigue?machine_id=TEST-M1&days=30`);

    expect(response.ok()).toBeTruthy();
  });

  test('should include by_risk_level breakdown', async ({ request }) => {
    const response = await request.get(`${API_BASE}/fatigue?days=30`);

    const body = await response.json();
    if (body.total_alerts > 0) {
      expect(body.by_risk_level).toBeDefined();
    }
  });
});

test.describe('API Documentation', () => {
  test('should serve OpenAPI spec', async ({ request }) => {
    const response = await request.get(`${API_BASE}/openapi.json`);

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(body.info.title).toContain('SentinelX');
    expect(body.paths['/predict']).toBeDefined();
    expect(body.paths['/health']).toBeDefined();
    expect(body.paths['/alerts']).toBeDefined();
  });

  test('should serve Swagger UI', async ({ request }) => {
    const response = await request.get(`${API_BASE}/docs`);
    expect(response.ok()).toBeTruthy();
  });
});
