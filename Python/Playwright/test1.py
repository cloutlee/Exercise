from playwright.sync_api import sync_playwright


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    page.goto("https://tw.dictionary.search.yahoo.com/")

    page.wait_for_load_state("networkidle")

    page.evaluate("window.scrollBy(0, 600)")

    page.screenshot(path="1.png")

    browser.close()
