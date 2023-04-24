import textwrap

def test_read(read_set, qname):
    agrees_with_expected, layout, expected = read_set.compare_to_expected(qname)

    diagnostic_message = textwrap.dedent(f'''
        set name: {read_set.name}
        query name: {qname}
        expected: ({expected["category"]}, {expected["subcategory"]})
        actual: ({layout.category}, {layout.subcategory})
        note: {expected["note"]}
    ''')

    assert agrees_with_expected, diagnostic_message