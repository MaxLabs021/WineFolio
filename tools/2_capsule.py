import sys, numpy as np, cv2
from PIL import Image
src, dst = sys.argv[1], sys.argv[2]
im = np.array(Image.open(src).convert("RGBA")).astype(np.float32)
a = im[:,:,3]; H, W = a.shape
row0 = np.where(a[0] > 100)[0]
if len(row0) == 0:
    Image.fromarray(im.astype(np.uint8)).save(dst); print("bouteille entière"); sys.exit()
x0, x1 = row0.min(), row0.max(); w = x1-x0+1

# 1. Hauteur de la capsule visible : on descend tant que c'est rouge/coloré (pas le verre sombre)
def is_cap(y):
    r = im[y, x0+w//5:x1-w//5, :3]; return np.median(r.max(axis=1)) > 70
cap_h = 0
while cap_h < H//4 and is_cap(cap_h): cap_h += 1
# 2. Opacité pleine dans la capsule (le détourage la rendait semi-transparente)
for y in range(cap_h):
    xs = np.where(a[y] > 60)[0]
    if len(xs): im[y, xs.min():xs.max()+1, 3] = 255
# 3. Zone "propre" (sans texte doré) dans la capsule pour échantillonner la texture
def gold_count(y):
    r = im[y, x0+w//8:x1-w//8, :3]; return ((r[:,1] > r[:,0]*0.55) & (r[:,0] > 120)).sum()
clean = [y for y in range(cap_h) if gold_count(y) == 0]
# on part du haut : premier bloc propre continu
start = clean[0] if clean else 0
blk = [start]
for y in clean[1:]:
    if y == blk[-1]+1: blk.append(y)
    else: break
rows = im[0:max(cap_h-8,10), x0:x1+1, :3]
gold = (rows[:,:,1] > rows[:,:,0]*0.55) & (rows[:,:,0] > 110)
gold = cv2.dilate(gold.astype(np.uint8), np.ones((5,5),np.uint8)).astype(bool)
masked = np.where(gold[:,:,None], np.nan, rows)
prof = np.nanmedian(masked, axis=0)
prof = np.where(np.isnan(prof), np.nanmedian(prof,axis=0), prof)
prof = cv2.GaussianBlur(prof[None].astype(np.float32),(0,0),sigmaX=2)[0]
noise = np.nan_to_num(masked - prof[None], nan=0.0)
noise = noise[~gold.all(axis=1)]
g_full = [((im[y,x0+12:x1-11,1] > im[y,x0+12:x1-11,0]*0.55) & (im[y,x0+12:x1-11,0] > 110)).sum() for y in range(cap_h)]
cover = 0
for y in range(cap_h-8):
    if all(g <= 1 for g in g_full[y:y+6]): cover = y + 3; break                      # on recouvre le texte coupé au-dessus de la zone propre
E = int(w*0.95) - cover; E = max(E, int(w*0.5)); R = max(4, int(w*0.07))
P = E
out = np.zeros((H+P, W, 4), np.float32); out[P:] = im
total = E + cover
ext = np.zeros((total, w, 4), np.float32)
rng = np.random.default_rng(3)
for y in range(total): ext[y,:,:3] = prof + noise[rng.integers(0,len(noise))]*0.9
yy = np.linspace(0,1,total)[:,None,None]; ext[:,:,:3] *= (0.95+0.05*yy)
ext[R:R+3,:,:3] *= 0.86
m = np.zeros((total*4, w*4), np.uint8)
cv2.rectangle(m,(0,R*4),(w*4-1,total*4-1),255,-1); cv2.ellipse(m,(w*2,R*4),(w*2-1,R*4),0,180,360,255,-1)
ext[:,:,3] = cv2.resize(m,(w,total),interpolation=cv2.INTER_AREA)
top = np.zeros((total,w),np.uint8); cv2.ellipse(top,(w//2,R),(w//2-2,R-1),0,0,360,1,-1)
ext[:,:,:3] = np.where(top[:,:,None]>0, np.clip(ext[:,:,:3]*1.1,0,255), ext[:,:,:3])
# raccord progressif sur 6 rangs dans la zone propre
region = out[0:total, x0:x1+1]
for y in range(total):
    t = np.clip((y-(total-6))/6, 0, 1)   # 0 = plein reconstruit, 1 = original
    if y < P: region[y] = ext[y]
    else:
        region[y,:,:3] = ext[y,:,:3]*(1-t) + region[y,:,:3]*t; region[y,:,3] = 255
out[0:total, x0:x1+1] = region
Image.fromarray(np.clip(out,0,255).astype(np.uint8)).save(dst); print("ok capsule", cap_h, "propre", blk[0], blk[-1], "ajout", E)
