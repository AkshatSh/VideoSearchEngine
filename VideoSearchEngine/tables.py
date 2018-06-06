from flask_table import Table, Col
 
class Results(Table):
    id = Col('Id', show=False)
    name = Col('name', column_html_attrs={'class':'table_col'})
    url = Col('url', column_html_attrs={'class':'table_col'})
    