import uvicorn

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.encoders import jsonable_encoder

from utils.preprocess import cleanComment, caseFolding, replaceSlangWords, tokenize, stopwordRemoval, stemming

app = FastAPI()

origins = ["*"] # list array allowed access origin

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def hello():
    content = jsonable_encoder({"message": "Hello World!"})
    return JSONResponse(
        status_code=200,
        content=content
    )

class PredictRequest(BaseModel):
    lang: str
    text: str

# load model and vect train
vectorizer = joblib.load('./vect/vect_train.pkl')
model = joblib.load('./model/knn_model.pkl')

@app.post("/predict")
def predict(payload: PredictRequest):
    try:
        lang = payload.lang
        text = payload.text

        # Convert text to DataFrame
        df_text = pd.DataFrame([text], columns=['text'])

        # Preprocess
        df_text['cleaned_text'] = df_text['text'].apply(cleanComment)
        df_text['folded_text'] = df_text['cleaned_text'].apply(caseFolding)
        df_text['slang_replaced_text'] = df_text['folded_text'].apply(lambda x: replaceSlangWords(x, lang))
        df_text['tokenized_text'] = df_text['slang_replaced_text'].apply(tokenize)
        df_text['stopword_removed_text'] = df_text['tokenized_text'].apply(lambda x: stopwordRemoval(x, lang))
        df_text['stemmed_text'] = df_text['stopword_removed_text'].apply(lambda x: stemming(x, lang))

        # Debug 
        print(f"cleaned_text: {df_text['cleaned_text'][0]}")
        print(f"folded_text: {df_text['folded_text'][0]}")
        print(f"slang_replaced_text: {df_text['slang_replaced_text'][0]}")
        print(f"tokenized_text: {df_text['tokenized_text'][0]}")
        print(f"stopword_removed_text: {df_text['stopword_removed_text'][0]}")
        print(f"stemmed_text: {df_text['stemmed_text'][0]}")

        # tf-idf weighting
        try:
            text_vector = vectorizer.transform(df_text['stemmed_text'])
            df_text_vector = pd.DataFrame(text_vector.toarray(), columns=vectorizer.get_feature_names_out())
            df_text_vector = df_text_vector.drop('sentimen', axis=1)
            print(f"text_vector: {text_vector}")
        except Exception as e:
            print(f"Error in vectorizing text: {str(e)}")
            return JSONResponse(status_code=500, content=jsonable_encoder({
                "status": "failed",
                "error": "Transform function failed"
            }))

        # predict
        try:
            prediction = model.predict(df_text_vector)
            print(f"prediction: {prediction}")
        except Exception as e:
            print(f"Error in predicting: {str(e)}")
            return JSONResponse(status_code=500, content=jsonable_encoder({
                "status": "failed",
                "error": "Prediction function failed"
            }))

        return JSONResponse(
            status_code=200,
            content=jsonable_encoder({
                "status": "success",
                "lang_text": str(lang),
                "original_text": str(text),
                "prediction": prediction[0]
            })
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=jsonable_encoder({
                "status": "failed",
                "error": str(e)
            })
        )
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)