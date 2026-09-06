import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  testMatch: "*.spec.ts",
  timeout: 60_000,
  workers: 2,
  use: { baseURL: "http://127.0.0.1:4173" },
  webServer: { command: "node tests/server.mjs", url: "http://127.0.0.1:4173", reuseExistingServer: false },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "firefox", use: { ...devices["Desktop Firefox"] } },
    { name: "iphone-webkit", use: { ...devices["iPhone 13"], defaultBrowserType: "webkit" } },
    { name: "android-chromium", use: { ...devices["Pixel 7"] } },
  ],
});
