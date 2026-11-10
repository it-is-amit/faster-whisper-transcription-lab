from playwright.sync_api import sync_playwright

PROFILE_PATH = r"S:\Work\ramayana-transcriber\chrome-profile"

with sync_playwright() as p:

    browser = p.chromium.launch_persistent_context(
        user_data_dir=PROFILE_PATH,
        channel="chrome",
        headless=False
    )

    page = browser.new_page()

    page.goto(
    "https://play.google.com/books/reader?id=SWhtCwAAQBAJ&pg=GBS.PA5&hl=en_GB"
    )

    print("Google Books opened.")

    input("Book open karke reader par jao, phir Enter press karo...")

    print("\nPage title:", page.title())

    print("\nVisible page text:")
    text = page.locator("body").inner_text()

    print(text[:10000])

    input("\nInspection complete. Enter press karke close karo...")

    browser.close()