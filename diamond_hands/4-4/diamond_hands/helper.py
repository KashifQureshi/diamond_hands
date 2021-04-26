dash_header = ''' 
<div class = "top">
    <div class="header">
        <h2>Group 2 Diamond Hands - /r/wallstreetbets Sentiment Analysis</h2>
        <p></p>
    </div>
    <div id="navbar" >
        <a id = "home" class="$home" href="/">Home</a>
        <a id = "data" class="$data" href="/data">Data</a>
        <a id = "analytics" class="$analytics" href="/analytics">Analytics</a>
        <a id = "about" class="$aboutus" href="/aboutus">About Us</a>
    </div>
    </div>
    '''

# Operators
operators = [['ge ', '>='], ['le ', '<='], ['lt ', '<'], ['gt ', '>'], ['ne ', '!='], ['eq ', '='], ['contains '],
             ['datestartswith ']]


# Filtering the table
def split_filter_part(filter_part):
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find('{') + 1: name_part.rfind('}')]
                value_part = value_part.strip()
                v0 = value_part[0]
                if v0 == value_part[-1] and v0 in ("'", '"', '`'):
                    value = value_part[1: -1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part
                # word operators need spaces after them in the filter string,
                # but we don't want these later
                return name, operator_type[0].strip(), value
    return [None] * 3


def update_filtered_table(page_current, page_size, sort_by, filter, dff):
    filtering_expressions = filter.split(' && ')
    for filter_part in filtering_expressions:
        col_name, operator, filter_value = split_filter_part(filter_part)
        if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            # these operators match pandas series operator method names
            dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
        elif operator == 'contains':
            dff = dff.loc[dff[col_name].str.contains(filter_value)]
        elif operator == 'datestartswith':
            # this is a simplification of the front-end filtering logic,
            # only works with complete fields in standard format
            dff = dff.loc[dff[col_name].str.startswith(filter_value)]
    if len(sort_by):
        dff = dff.sort_values([col['column_id'] for col in sort_by],
                              ascending=[col['direction'] == 'asc' for col in sort_by], inplace=False)
    page = page_current
    size = page_size
    return dff.iloc[page * size: (page + 1) * size].to_dict('records')