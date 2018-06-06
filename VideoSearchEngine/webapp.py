from flask import Flask, render_template, request, redirect, flash
from forms import VideoSearchForm, AddVideoForm
from tables import Results
from page_rank import rank_pages
from database_utils import upload_new_summary, get_all_data, get_url, get_id_from_name

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    search = VideoSearchForm(request.form)
    if request.method == 'POST':
        return search_results(search)
    
    return render_template('index.html', form=search)

@app.route('/results')
def search_results(search):
    results = []
    data = get_all_data()
    search_string = search.data['search']
    scores = rank_pages(data, search_string)

    results = []

    for video in scores:
        vid_id = str(get_id_from_name(video))
        results.append({'name': video, 'url': get_url(vid_id)})
  
    table = Results(results, html_attrs={'class':'results_table'})
    table.border = True
    return render_template('results.html', table=table, form=search)

if __name__ == '__main__':
    app.run()