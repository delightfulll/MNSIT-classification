from fastapi import FastAPI, HTTPException, UploadFile, File
import torch
from torchvision import transforms
import uvicorn

app = FastAPI()
model = torch.load("models/MNIST_model.pt", map_location= 'cpu')
transform = transforms.Compose([transforms.Grayscale(), transforms.Resize((28, 28)), transforms.ToTensor])
#api to get the result from out saved model
@app.post("/predict/{photoid}")
async def predict_photo(file: UploadFile = File(...)):
    if file.filename == "":
        return HTTPException(status_code= 404, detail= "No file selected")
    
    #apply our transformations for the photo
    photo = transform(file)
    with torch.no_grad():
        #move to gpu if exists
        #make prediction
        outputs = model(photo)
        #since using log softmax, convert each output to exp
        pred_label_proba = torch.exp(outputs)
        
        _, pred_labels = torch.max(pred_label_proba, dim=1)
    
    return {"Prediction:" : pred_label_proba}


#run our server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

