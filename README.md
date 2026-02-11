# chatbot-tensorflow-v2.0

This is a chatbot which works with tensorflow 2.1 and higher. 
It also saves wrong answers with predicted category in a text file named as 'exceptions.txt'

Requirements:<br>
-Tensorflow 2.0 or higher<br>
-Nltk<br>
-Punkt from nltk &nbsp;&nbsp;&nbsp;&nbsp; (nltk.download('punkt'))
-Punkt from nltk &nbsp;&nbsp;&nbsp;&nbsp; (nltk.download('punkt_tab'))

python -m venv .venv
.\.venv\Scripts\activate
pip install numpy
pip install tensorflow
pip install nltk

Once I had both of the above installed, you should also see me setup the Punkt from nltk in the terminal like:

python.exe
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

Make sure to remove the model.h5/model.keras file when you change the intents.json file

Changes from the original
Line 87: model.save('model.h5') → model.save('model.keras') — saves in native Keras format instead of legacy HDF5
Line 90: keras.models.load_model('model.h5') → keras.models.load_model('model.keras') — loads the new format
Line 121: model.predict([bag_of_words(inp, words)]) → model.predict(bag_of_words(inp, words)) — removed the extra list wrapper since bag_of_words() already returns a 2D array with shape (1, N)
TF_CPP_MIN_LOG_LEVEL=2 — Suppresses TensorFlow info (I) and warning (W) messages, only showing errors. (0=all, 1=no info, 2=no info/warnings, 3=errors only)
TF_ENABLE_ONEDNN_OPTS=0 — Disables oneDNN custom operations, which removes the floating-point round-off warning
