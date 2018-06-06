from flask_table import Table, Col
 
class Results(Table):
    id = Col('Id', show=False)
    name = Col('Name', column_html_attrs={'class':'table_col name'}, th_html_attrs={'class':'table_header'})
    summary = Col('Extractive Summary', column_html_attrs={'class':'table_col summary'}, td_html_attrs={'class':'sumDescrip'}, th_html_attrs={'class':'table_header'})
    url = Col('URL', column_html_attrs={'class':'table_col url'}, th_html_attrs={'class':'table_header'})
    