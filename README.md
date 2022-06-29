
# MC NLP Project

A project to analyse and report on common topics in text data using Topic Modelling and a Streamlit dashboard.

## Run Locally

Clone the project

```bash
  git clone https://github.com/wgdsu/MC_NLP
```

Go to the project directory

```bash
  cd MC_NLP
```

Please then check dependencies. A requirements.txt file will be added for this.

Start the application with a command line argument to decide which model you would like to build and an argument to configure Welsh language support on your dataset.

(Supported models are *'nmf'* and *'lda'*)
(1 = Welsh language support 0 = no Welsh language support).

```bash
  python3 build_model.py nmf 0
```

Once this code has run you can start up the dashboard:

```bash
  streamlit run dashboard.py

```
To use an embedding model in assessing topic similarity you will need to install the pretrained Spacy model

```bash
python -m spacy download en_core_web_md
```

## Welsh Language Support
This work is made easier thanks to the list of Welsh language stop words compiled here: https://github.com/techiaith/ataleiriau
