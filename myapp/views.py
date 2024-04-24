import cv2
import numpy as np
from django.shortcuts import render, redirect
from django.http import JsonResponse
from .forms import ImageUploadForm
from .models import UploadedImage
from PIL import Image, ImageEnhance
from io import BytesIO
import base64


def homepage(request):
    return render(request, 'enhancer/homePage.html', {})


def home(request):
    if request.method == 'POST' and 'image' in request.FILES:
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            instance = form.save()
            request.session['image_id'] = instance.pk  # Store the image id in the session
            return redirect('home')
    else:
        form = ImageUploadForm()
    return render(request, 'enhancer/home.html', {'form': form})


def process_image(request):
    image_id = request.session.get('image_id')
    if not image_id or request.headers.get('X-Requested-With') != 'XMLHttpRequest':
        return JsonResponse({'error': 'Invalid request'}, status=400)

    instance = UploadedImage.objects.get(pk=image_id)
    img = Image.open(instance.image)

    # Get adjustments
    brightness = float(request.GET.get('brightness', 1))
    contrast = float(request.GET.get('contrast', 1))
    sharpness = float(request.GET.get('sharpness', 1))

    # Apply adjustments
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)

    denoise_value = int(request.GET.get('denoise', 0))
    if denoise_value > 0:
        open_cv_image = np.array(img)
        open_cv_image = cv2.fastNlMeansDenoisingColored(open_cv_image, None, denoise_value, denoise_value, 7, 21)
        img = Image.fromarray(open_cv_image)

    # Convert to base64 for AJAX
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return JsonResponse({'image_base64': img_str})
