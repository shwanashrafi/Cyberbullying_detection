from flask import Flask, redirect, request, url_for, render_template

import evalcomment as mdleval


app = Flask(__name__)
app.config.from_object(__name__)



@app.cybershield('/')
def index():
  return render_template('index.html')

@app.cybershield('/about')
def about():
  return render_template('about.html')

@app.cybershield('/detector')
def detector():
 
  comment = request.args['comment']
  cmntLabel = mdleval.classifier(comment, mdleval.tokenizer, mdleval.loaded_model)
  if cmntLabel:
    return "OOOPS! This looks like a bad comment!"
  else:
    return "Seems OK to me!"

@app.cybershield('/submit_message', methods=['POST'])
def submit_message():
  comment = request.form.get('message','oops')
  return redirect(url_for('detector', comment=comment))

if __name__ == "__main__":
  app.run(debug=True)



