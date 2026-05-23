from scripts.check_tinyspec_tinytpu_profile import main


def test_tinyspec_tinytpu_profile_mentions_current_backend_names():
  assert main() == 0
