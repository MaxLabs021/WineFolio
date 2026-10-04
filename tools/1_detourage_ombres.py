import numpy as np, cv2
from PIL import Image, ImageOps
from rembg import remove, new_session
img = ImageOps.exif_transpose(Image.open(__import__('sys').argv[1])).convert("RGB"); img.thumbnail((2000,2000))
rgb = np.array(img).astype(np.float32)
alpha = np.array(remove(img, session=new_session("isnet-general-use")))[:,:,3]

# 1. Masque des étiquettes : zones claires et peu saturées dans la bouteille
hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
cand = ((hsv[:,:,1] < 120) & (hsv[:,:,2] > 110) & (alpha > 128)).astype(np.uint8)
cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((25,25),np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(cand)
mask = np.zeros_like(cand)
for i in range(1,n):
    if st[i, cv2.CC_STAT_AREA] > 0.01*cand.size: mask[lab==i] = 1
# remplir les trous (texte)
cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros_like(mask); cv2.drawContours(mask, cnts, -1, 1, -1)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((41,41),np.uint8)); mask = cv2.erode(mask, np.ones((5,5),np.uint8))

# 2. Estimation de l'éclairage du papier (on efface le texte par filtre max, puis flou normalisé)
out = rgb.copy()
m = mask.astype(np.float32)
gray = rgb.mean(axis=2)
sat = hsv[:,:,1].astype(np.float32)
lab_sat = np.median(sat[mask>0]) if mask.any() else 0
paper = cv2.dilate(gray, np.ones((27,27),np.uint8))
bg = cv2.GaussianBlur(paper*m,(0,0),7)/(cv2.GaussianBlur(m,(0,0),7)+1e-4)
for it in range(3):
    pm = ((gray > bg*0.70) & (np.abs(sat-lab_sat) < 28) & (mask>0)).astype(np.float32)
    pm = cv2.erode(pm, np.ones((3,3),np.uint8))
    bgs = []
    for c in range(3):
        num = cv2.GaussianBlur(rgb[:,:,c]*pm,(0,0),2.5); den = cv2.GaussianBlur(pm,(0,0),2.5)+1e-4
        wide_n = cv2.GaussianBlur(rgb[:,:,c]*pm,(0,0),20); wide_d = cv2.GaussianBlur(pm,(0,0),20)+1e-4
        w = np.clip(den*3,0,1)
        bgs.append(w*num/den + (1-w)*wide_n/wide_d)
    bg = np.mean(bgs,axis=0)
ratio = pm.sum()/max(mask.sum(),1)
print("part de papier uni sur l'étiquette:", round(float(ratio),2))
if ratio < 0.5:
    bgs = None  # étiquette illustrée : on ne touche pas aux couleurs
# couleur cible = couleur du papier bien éclairé (on garde le crème d'une étiquette crème)
sel = pm > 0
for c in (range(3) if bgs is not None else []):
    target = np.percentile(rgb[:,:,c][sel], 90) if sel.sum() > 500 else 240
    target = min(target*1.03, 248)
    out[:,:,c] = np.where(mask>0, np.clip(rgb[:,:,c]/np.maximum(bgs[c],1)*target,0,255), rgb[:,:,c])
# 3. Fondu doux sur les bords de l'étiquette
feather = cv2.GaussianBlur(m, (0,0), 3)[:,:,None]
res = (out*feather + rgb*(1-feather)).astype(np.uint8)
Image.fromarray(np.dstack([res, alpha])).save(__import__('sys').argv[2])

print("ok")
