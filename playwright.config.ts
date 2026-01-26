import { defineConfig, devices } from '@playwright/test';

/**
 * SentinelX Playwright Configuration
 * ==================================
 *
 * Run tests:
 *   npx playwright test
 *   npx playwright test --ui          # Interactive mode
 *   npx playwright test --headed      # See browser
 *   npx playwright test --project=chromium
 *
 * Prerequisites:
 *   docker-compose up -d  # Start services first
 */

export default defineConfig({
  testDir: './tests/e2e',

  // Run tests in parallel
  fullyParallel: true,

  // Fail the build on CI if you accidentally left test.only
  forbidOnly: !!process.env.CI,

  // Retry on CI only
  retries: process.env.CI ? 2 : 0,

  // Opt out of parallel tests on CI
  workers: process.env.CI ? 1 : undefined,

  // Reporter configuration - saves results in multiple formats
  reporter: [
    // Interactive HTML report (open with: npx playwright show-report)
    ['html', { outputFolder: 'playwright-report', open: 'never' }],

    // JSON report for programmatic access
    ['json', { outputFile: 'test-results/results.json' }],

    // JUnit XML for CI/CD (GitHub Actions, Jenkins, etc.)
    ['junit', { outputFile: 'test-results/junit.xml' }],

    // Console output during test run
    ['list']
  ],

  // Output directory for traces, screenshots, videos
  outputDir: 'test-results/artifacts',

  // Shared settings for all projects
  use: {
    // Base URL for API tests
    baseURL: 'http://localhost:8000',

    // Collect trace when retrying the failed test
    trace: 'on-first-retry',

    // Screenshot on failure
    screenshot: 'only-on-failure',

    // Video on failure
    video: 'retain-on-failure',
  },

  // Configure projects for major browsers
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },
    // Mobile viewports
    {
      name: 'Mobile Chrome',
      use: { ...devices['Pixel 5'] },
    },
  ],

  // Timeout settings
  timeout: 60000,
  expect: {
    timeout: 10000
  },

  // Web server to start before tests (optional - use if not running docker-compose)
  // webServer: [
  //   {
  //     command: 'docker-compose up',
  //     url: 'http://localhost:8000/health',
  //     reuseExistingServer: !process.env.CI,
  //     timeout: 120000,
  //   }
  // ],
});
