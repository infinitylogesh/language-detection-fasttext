import fastText
import sys
reload(sys)  
sys.setdefaultencoding('UTF8') # default encoding to utf-8

model = fastText.load_model("bin/lid.176.ftz")

def predict_lang(model,texts): return model.predict(texts,k=1)

test_texts = ["மதியும் மடந்தை முகனு மறியா பதியிற் கலங்கிய மீன்.","Incapaz de distinguir la luna y la cara de esta chica,Las estrellas se ponen nerviosas en el cielo.","Unable to tell apart the moon and this girl's face,Stars are flustered up in the sky."]
for text in test_texts:
    prediction = predict_lang(model,text)
    label = prediction[0][0].split("__label__")[1]
    print "{} is {}".format(text,label)

