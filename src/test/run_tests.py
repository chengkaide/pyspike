"""Run the double spike toolbox test suite.

The tests are written for pytest, but this runner also works when pytest is not
installed (it supplies a tiny stand-in for ``pytest.mark.parametrize``).  That
keeps the package dependency free for people who just want to check their
install.

Usage::

    python -m test.run_tests          # from src/
    python src/test/run_tests.py      # from the repository root
"""

import importlib.util
import os
import sys
import traceback
import types


def _install_pytest_shim():
    """Provide a minimal pytest module if the real one is missing."""
    try:
        found = importlib.util.find_spec("pytest")
    except (ImportError, ValueError):
        found = None
    if found is not None:
        return False

    mod = types.ModuleType("pytest")

    class _Mark:
        def parametrize(self, names, values):
            def decorator(func):
                func._parametrize = (names, values)
                return func

            return decorator

        def __getattr__(self, name):  # e.g. pytest.mark.slow
            def decorator(func):
                return func

            return decorator

    mod.mark = _Mark()
    sys.modules["pytest"] = mod
    return True


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    shimmed = _install_pytest_shim()
    if shimmed:
        print("pytest not installed: using the built in mini runner\n")

    passed = failed = 0
    failures = []
    for fname in sorted(os.listdir(here)):
        if not (fname.startswith("test_") and fname.endswith(".py")):
            continue
        path = os.path.join(here, fname)
        spec = importlib.util.spec_from_file_location(fname[:-3], path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[fname[:-3]] = module
        spec.loader.exec_module(module)

        for name in sorted(vars(module)):
            func = getattr(module, name)
            if not (name.startswith("test_") and callable(func)):
                continue
            cases = [()]
            if hasattr(func, "_parametrize"):
                names, values = func._parametrize
                cases = [tuple(v) if isinstance(v, (tuple, list)) else (v,) for v in values]
            for case in cases:
                label = f"{fname}::{name}{case if case else ''}"
                try:
                    func(*case)
                    passed += 1
                    print(f"PASS  {label}")
                except Exception:  # noqa: BLE001
                    failed += 1
                    failures.append((label, traceback.format_exc()))
                    print(f"FAIL  {label}")

    print()
    for label, tb in failures:
        print("=" * 70)
        print("FAILED:", label)
        print(tb)
    print(f"{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
