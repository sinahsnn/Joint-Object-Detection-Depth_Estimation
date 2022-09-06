import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from DenseDepth import load_depths, depth_predict
from matplotlib.colors import TABLEAU_COLORS
from collections import Counter


def load_models(yolo_path, yolo_model, indoor_path, outdoor_path):

    indoor, outdoor = load_depths(indoor_path, outdoor_path)
    yolo = torch.hub.load(yolo_path, 'custom', path=yolo_model, source='local')
	
    return yolo, indoor, outdoor


def process(yolo, indoor, outdoor, image, check_func, is_indoor=True):

    result = yolo(image)
    depths = depth_predict(indoor, image, 10) if is_indoor else depth_predict(outdoor, image, 80)

    df = result.pandas().xyxy[0]

    counts = Counter(df['name'])
    names = sorted(counts, key=counts.get, reverse=True)
    colors = dict(zip(names, list(TABLEAU_COLORS.values())[:len(names)]))

    pil_image = Image.fromarray(np.uint8(image))
    font = ImageFont.truetype('arial.ttf', 12)

    objects = []

    for index, row in df.iterrows():
        x0 = round(row['xmin'])
        y0 = round(row['ymin'])
        x1 = round(row['xmax'])
        y1 = round(row['ymax'])
        w = x1 - x0
        h = y1 - y0
        n = row['name']
        c = row['confidence']

        d = np.mean(depths[round(y0 + (h * 0.3)): round(y1 - (h * 0.3)), round(x0 + (w * 0.3)): round(x1 - (w * 0.3))])
        if not check_func(d):
            continue

        draw = ImageDraw.Draw(pil_image)
        draw.rectangle((x0, y0, x1, y1), outline=colors[n])
        text = ' ' + n + ' ' + str(counts[n]) + ' '
        w, h = font.getsize(text)
        draw.rectangle((x0, y0, x0 + w, y0 - h), fill=colors[n], width=3)
        draw.text((x0, y0), text, fill=(255, 255, 255), font=font, anchor='lb')

        objects.append({'name': n + ' ' + str(counts[n]), 'color': colors[n], 'confidence': c, 'depth': d})
        counts[n] -= 1

    objects = sorted(objects, key=lambda x: x['name'])

    return pil_image, objects
