from dash import dcc, html, Input, Output, callback

layout = html.Div([
    html.H3('Page doc'),

    dcc.Markdown('''
        # Debuggage
        Une Une fois les requirements installés, pour débugger, utilisez votre terminal :
        ```bash
        source ./venv/bin/activate
        which python3
        which pip
        # pip install -r yolov5/requirements.txt # Normalement déjà installés (pytorch / opencv / ...)
        python3 ./YOLO/YOLO.py --origin ./YOLO/Testset --weights ./uploads/weights.pt --conf_thres 0.4 # Testez l'inférence
        sudo chmod -R 777 /var/www/html # Donner les droits admin au nouveaux fichiers
        ```
        ##### by Dorian VOYDIE

    ''')


])


