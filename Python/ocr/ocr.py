import pytesseract
from PIL import Image

img = Image.open('12.png')
img.show()
text = pytesseract.image_to_string(img, lang="chi_tra+eng")
print(text)