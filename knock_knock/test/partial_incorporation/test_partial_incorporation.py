import knock_knock.test.partial_incorporation

get_all_read_sets = knock_knock.test.partial_incorporation.get_all_read_sets

def test_read(read_set, read_name):
    matches, diagnostic_message = read_set.compare_to_expected(read_name)

    assert matches, diagnostic_message