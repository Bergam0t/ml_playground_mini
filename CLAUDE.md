# CLAUDE.md

## Running the app headlessly (streamlit.testing AppTest)

Headless runs need **both** of these, because two things resolve paths against different bases:

- **Give `AppTest.from_file` an absolute path**, e.g.
  `AppTest.from_file("C:/ml_playground_mini/streamlit_app.py")`. A relative path is resolved
  against the folder of the *test script*, not the working directory, so a test script kept in a
  scratchpad or temp folder fails with `FileNotFoundError: AppTest script not found at <scratchpad>/streamlit_app.py`.
- **Run it from the repo root.** The app itself opens `style.css` and
  `resources/fonts/...` relative to the *working directory* (streamlit_app.py, near the top), so
  it breaks if the test is started from anywhere else.

Also:

- **Use the project's virtual environment**, `.venv/Scripts/python.exe`. It has the pinned
  Streamlit (1.66.1). The system Python has an older Streamlit that fails on
  `st.html(..., unsafe_allow_javascript=True)`.
- **Put the repo on `sys.path`** (`sys.path.insert(0, "C:/ml_playground_mini")`) at the top of a
  test script kept outside the repo. Otherwise the app's own imports (`import calls`) fail.

A first `at.run()` takes about 4 seconds; use `default_timeout=60` or so. If a headless run
hangs with no output at all, check the command itself (e.g. something reading stdin) before
blaming the app: the app's only `time.sleep` loop is the model "race" on the first tab, which
runs only after its run button is clicked.
