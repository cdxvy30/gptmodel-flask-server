import numpy as np
import torch
import torchvision
import json
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from flask import Flask, jsonify, request
import io
import clip

def get_model():
    model_path = f'./model_final.pth'
    num_classes = 7
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    checkpoint = torch.load(model_path) if torch.cuda.is_available() else torch.load(model_path, map_location='cpu')
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


application = Flask(__name__)

def object_detection(image):
    # model = get_model()
    
    ## transform image to tensor
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)

    # predict
    model.eval()
    with torch.no_grad():
        prediction = model(image_tensor.to(device))[0]

    output = {}
    output['boxes'] = prediction['boxes'].cpu().numpy().tolist()
    output['scores'] = prediction['scores'].cpu().numpy().tolist()
    output['labels'] = []
    prediction_label_ids = prediction['labels'].cpu().numpy().tolist()
    for id in prediction_label_ids:
        output['labels'].append(get_label(str(id)))

    return output

def get_label(id):
    f = open('labels.json')
    data = json.load(f)
    labels = data["labels"]
    # Closing file
    f.close()
    return labels[id]

def clip_classification(image):
    image = torch.tensor(preprocess(image)).to(device).unsqueeze(0)
    prediction = {}

    for key in ['caption_type', 'violation_type']:
        type = clip.tokenize(type_dict[key]).to(device)

        logits_per_image, _ = clip_model(image, type)
        prediction[key] = type_dict[key][torch.argmax(logits_per_image, dim=1).item()]

    return prediction['caption_type'], prediction['violation_type']

@application.route('/predict', methods=['POST'])
def predict():
    i = 0
    if request.method == 'POST':
        file = request.files['file']
        image_extensions=['ras', 'xwd', 'bmp', 'jpe', 'jpg', 'jpeg', 'xpm', 'ief', 'pbm', 'tif', 'gif', 'ppm', 'xbm', 'tiff', 'rgb', 'pgm', 'png', 'pnm']
        if file.filename.split('.')[1] not in image_extensions:
            return jsonify('Please upload an appropriate image file')

        saveLocation = file.filename
        file.save(saveLocation)
        image = Image.open(saveLocation)

        prediction = object_detection(image)
        caption_type, violation_type = clip_classification(image)

        return jsonify({"boxes": prediction['boxes'], 
            "labels": prediction['labels'], 
            "scores": prediction['scores'], 
            "caption_type": caption_type,
            "violation_type": violation_type,})

@application.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. """
    return jsonify({"response": __name__})

@application.route("/")
def home():
    return "Hello, World!"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
clip_model, preprocess = clip.load("ViT-B/32", device=device)
model_path = f'clip_balance_comb2_699_comb9_99.pt'
with open(model_path, 'rb') as opened_file: 
    clip_model.load_state_dict(torch.load(opened_file, map_location="cpu"))

model = get_model()

type_dict = {
    'caption_type': ['violation', 'status'],
    'violation_type': ['墜落', '機械', '物料', '感電', '防護具', '穿刺', '爆炸', '工作場所', '搬運']
}

if __name__ == "__main__":
    application.run(host='0.0.0.0', port=8000, debug=True)
