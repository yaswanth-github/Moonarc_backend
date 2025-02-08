## To Run Flash App
**- BackEnd Run**
```bash
python app.py
```

**- With File**

```bash
curl -X POST -F "file=@/Users/mac/Documents/MoonArc/Tested_images/nm.jpeg" http://127.0.0.1:5001/predict
```


```bash
curl -X POST -F "file=@/Users/mac/Documents/MoonArc/Tested_images/nm.jpeg" https://moonarc-backend.onrender.com/predict
```

2. Activate the Virtual Environment
After the virtual environment is created, activate it with:

```bash
source venv/bin/activate
```

3. Install Dependencies
Once activated, install the required Python libraries:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```