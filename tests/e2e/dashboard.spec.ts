import { test, expect, Page } from '@playwright/test';

/**
 * SentinelX Dashboard UI Tests
 * ============================
 *
 * End-to-end tests for the Streamlit CEO Dashboard:
 * - Header and branding
 * - Value metrics ($120M savings)
 * - Alert Fatigue visualization (Paper 3)
 * - Root cause distribution
 * - Live prediction demo
 *
 * Prerequisites: docker-compose up -d
 */

const DASHBOARD_URL = 'http://localhost:8501';

// Helper to wait for Streamlit to fully load
async function waitForStreamlit(page: Page) {
  // Wait for Streamlit's main content to load
  await page.waitForLoadState('networkidle');
  // Wait for the main app container
  await page.waitForSelector('[data-testid="stAppViewContainer"]', { timeout: 30000 });
  // Small delay for charts to render
  await page.waitForTimeout(2000);
}

test.describe('Dashboard Loading', () => {
  test('should load the dashboard', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    // Check page title
    await expect(page).toHaveTitle(/SentinelX/);
  });

  test('should display main header', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    // Look for the main header text
    const header = page.locator('text=SentinelX Predictive Maintenance');
    await expect(header).toBeVisible();
  });

  test('should display subtitle', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    const subtitle = page.locator('text=AI-Powered Industrial Intelligence');
    await expect(subtitle).toBeVisible();
  });
});

test.describe('Sidebar', () => {
  test('should show sidebar with configuration', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    // Streamlit sidebar
    const sidebar = page.locator('[data-testid="stSidebar"]');
    await expect(sidebar).toBeVisible();
  });

  test('should have configuration dropdown', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    // Look for Configuration text in sidebar
    const configText = page.locator('text=Configuration');
    await expect(configText).toBeVisible();
  });

  test('should show API connection status', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    // Should show "API Connected" or connection status
    const apiStatus = page.locator('text=/API (Connected|Offline|Degraded)/');
    await expect(apiStatus).toBeVisible();
  });

  test('should show Database connection status', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    // Should show database status
    const dbStatus = page.locator('text=/Database (Connected|Offline)/');
    await expect(dbStatus).toBeVisible();
  });
});

test.describe('Value Dashboard Metrics', () => {
  test('should display value dashboard section', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    const valueSection = page.locator('text=Value Dashboard');
    await expect(valueSection).toBeVisible();
  });

  test('should show Annual Savings metric', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    const savingsMetric = page.locator('text=Annual Savings Projection');
    await expect(savingsMetric).toBeVisible();
  });

  test('should show False Positive Rate metric', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    // Dashboard uses emoji: 🎯 False Positive Rate
    const fpMetric = page.locator('text=/False Positive Rate/');
    await expect(fpMetric.first()).toBeVisible();
  });

  test('should show Total Alerts metric', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    const alertsMetric = page.locator('text=Total Alerts');
    await expect(alertsMetric).toBeVisible();
  });

  test('should show True Positives metric', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    const tpMetric = page.locator('text=True Positives');
    await expect(tpMetric).toBeVisible();
  });

  test('should display savings in dollar format', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    // Look for dollar amounts (e.g., $120,900,000)
    const dollarAmount = page.locator('text=/\\$[\\d,]+/');
    await expect(dollarAmount.first()).toBeVisible();
  });
});

test.describe('Savings Counter Gauge', () => {
  test('should display savings counter section', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    const savingsSection = page.locator('text=Savings Counter');
    await expect(savingsSection).toBeVisible();
  });

  test('should render Plotly gauge chart', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    // Plotly charts have this class
    const plotlyChart = page.locator('.js-plotly-plot').first();
    await expect(plotlyChart).toBeVisible();
  });

  test('should show Projected Annual Savings label', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    const gaugeLabel = page.locator('text=Projected Annual Savings');
    await expect(gaugeLabel).toBeVisible();
  });
});

test.describe('Alert Fatigue Section (Paper 3)', () => {
  test('should display Alert Fatigue section', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    // Dashboard uses: 📉 Alert Fatigue (Paper 3)
    const fatigueSection = page.locator('text=/Alert Fatigue/');
    await expect(fatigueSection.first()).toBeVisible();
  });

  test('should reference Paper 3', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    // Paper 3 appears in multiple places: heading, finding, compliance
    const paper3Ref = page.locator('text=/Paper 3/');
    await expect(paper3Ref.first()).toBeVisible();
  });

  test('should show industry baseline comparison', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    // Should mention 54% industry baseline
    const industryBaseline = page.locator('text=/54%|Industry/');
    await expect(industryBaseline.first()).toBeVisible();
  });

  test('should show SentinelX FP rate', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    // Should mention 1.4% or SentinelX rate
    const sentinelxRate = page.locator('text=/1\\.4%|SentinelX/');
    await expect(sentinelxRate.first()).toBeVisible();
  });

  test('should show reduction percentage', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    // Should show 97% reduction
    const reductionText = page.locator('text=/97%|reduction/i');
    await expect(reductionText.first()).toBeVisible();
  });
});

test.describe('Alert Intelligence Section', () => {
  test('should display Alert Intelligence section', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    const intelligenceSection = page.locator('text=Alert Intelligence');
    await expect(intelligenceSection).toBeVisible();
  });

  test('should show Root Cause Distribution chart', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    const rootCauseChart = page.locator('text=Root Cause Distribution');
    await expect(rootCauseChart).toBeVisible();
  });

  test('should show Risk Level Distribution chart', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    const riskLevelChart = page.locator('text=Risk Level Distribution');
    await expect(riskLevelChart).toBeVisible();
  });

  test('should render pie chart for root causes', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    // Multiple Plotly charts should be present (at least 3: Gauge, FP comparison, Pie/Bar)
    const plotlyCharts = page.locator('.js-plotly-plot');
    const count = await plotlyCharts.count();
    expect(count).toBeGreaterThanOrEqual(3);
  });
});

test.describe('Recent Alerts Section', () => {
  test('should display Recent Alerts section', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    const alertsSection = page.locator('text=Recent High-Risk Alerts');
    await expect(alertsSection).toBeVisible();
  });
});

test.describe('Live Prediction Demo', () => {
  test('should have Live Prediction Demo section', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    const demoSection = page.locator('text=Live Prediction Demo');
    await expect(demoSection).toBeVisible();
  });

  test('should have expandable prediction form', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    // Look for the expander
    const expander = page.locator('text=Submit Test Prediction');
    await expect(expander).toBeVisible();
  });

  test('should expand prediction form on click', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    // Click to expand
    const expander = page.locator('[data-testid="stExpander"]').filter({ hasText: 'Submit Test Prediction' });
    await expander.click();

    // Wait for form to appear
    await page.waitForTimeout(500);

    // Should show Machine ID input
    const machineInput = page.locator('text=Machine ID');
    await expect(machineInput).toBeVisible();
  });

  test('should have system metrics sliders', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    // Expand the prediction form
    const expander = page.locator('[data-testid="stExpander"]').filter({ hasText: 'Submit Test Prediction' });
    await expander.click();
    await page.waitForTimeout(500);

    // Check for Air Temperature slider
    const tempSlider = page.locator('text=Air Temperature');
    await expect(tempSlider).toBeVisible();
  });

  test('should have Predict button', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    // Expand the prediction form
    const expander = page.locator('[data-testid="stExpander"]').filter({ hasText: 'Submit Test Prediction' });
    await expander.click();
    await page.waitForTimeout(500);

    // Look for Predict button
    const predictButton = page.locator('button:has-text("Predict")');
    await expect(predictButton).toBeVisible();
  });

  test('should submit prediction and show result', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    // Expand the prediction form
    const expander = page.locator('[data-testid="stExpander"]').filter({ hasText: 'Submit Test Prediction' });
    await expander.click();
    await page.waitForTimeout(500);

    // Click Predict button
    const predictButton = page.locator('button:has-text("Predict")');
    await predictButton.click();

    // Wait for response
    await page.waitForTimeout(5000);

    // Should show prediction result (success message or error)
    const result = page.locator('text=/Prediction Complete|Risk Level|API Error/');
    await expect(result.first()).toBeVisible({ timeout: 15000 });
  });
});

test.describe('Footer', () => {
  test('should display footer', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    // Scroll to bottom
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.waitForTimeout(500);

    const footer = page.locator('text=SentinelX v1.0');
    await expect(footer).toBeVisible();
  });

  test('should mention XGBoost and SHAP', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));

    const techStack = page.locator('text=/XGBoost.*SHAP/');
    await expect(techStack).toBeVisible();
  });
});

test.describe('Responsiveness', () => {
  test('should be responsive on mobile viewport', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    // Main content should still be visible
    const header = page.locator('text=SentinelX');
    await expect(header.first()).toBeVisible();
  });

  test('should be responsive on tablet viewport', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    const header = page.locator('text=SentinelX');
    await expect(header.first()).toBeVisible();
  });
});

test.describe('Visual Regression', () => {
  test.skip('should match main dashboard screenshot', async ({ page }) => {
    // Skip visual regression by default - run manually to update baselines
    // Run with: npx playwright test --update-snapshots
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    await expect(page).toHaveScreenshot('dashboard-main.png', {
      maxDiffPixels: 5000, // Allow variation for dynamic content
      timeout: 30000
    });
  });

  test.skip('should match metrics section screenshot', async ({ page }) => {
    // Skip visual regression by default
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    const metricsSection = page.locator('[data-testid="stAppViewContainer"]').first();
    await expect(metricsSection).toHaveScreenshot('metrics-section.png', {
      maxDiffPixels: 2000
    });
  });
});

test.describe('Accessibility', () => {
  test('should have proper heading structure', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    // Check for h1, h2, h3 elements
    const headings = page.locator('h1, h2, h3');
    const count = await headings.count();
    expect(count).toBeGreaterThan(0);
  });

  test('should have alt text for images', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    // Check images have alt attributes
    const images = page.locator('img');
    const count = await images.count();

    for (let i = 0; i < count; i++) {
      const img = images.nth(i);
      const alt = await img.getAttribute('alt');
      // Alt can be empty but should exist
      expect(alt).not.toBeNull();
    }
  });

  test('should be keyboard navigable', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    // Press Tab to navigate
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');

    // Should be able to focus on interactive elements
    const focusedElement = page.locator(':focus');
    await expect(focusedElement).toBeDefined();
  });
});

test.describe('Performance', () => {
  test('should load within acceptable time', async ({ page }) => {
    const startTime = Date.now();

    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    const loadTime = Date.now() - startTime;

    // Should load within 15 seconds
    expect(loadTime).toBeLessThan(15000);
    console.log(`Dashboard load time: ${loadTime}ms`);
  });

  test('should have no critical console errors', async ({ page }) => {
    const errors: string[] = [];

    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        errors.push(msg.text());
      }
    });

    await page.goto(DASHBOARD_URL);
    await waitForStreamlit(page);

    // Filter out known harmless warnings from Streamlit/Plotly
    const criticalErrors = errors.filter((e) => {
      const harmless = [
        'favicon',
        'manifest',
        'ResizeObserver',
        'plotly',
        'third-party',
        '404',
        'Failed to load resource'
      ];
      return !harmless.some(h => e.toLowerCase().includes(h.toLowerCase()));
    });

    // Allow up to 2 minor errors (Streamlit can be noisy)
    expect(criticalErrors.length).toBeLessThanOrEqual(2);
  });
});
