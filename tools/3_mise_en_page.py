import sys, numpy as np, cv2
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
b = Image.open(sys.argv[1]).convert("RGBA")
a = np.array(b)[:,:,3]; _,m = cv2.threshold(a,128,255,cv2.THRESH_BINARY)
n,lab,st,_ = cv2.connectedComponentsWithStats(m)
if n>1:
    keep=(lab==1+np.argmax(st[1:,cv2.CC_STAT_AREA])); arr=np.array(b); arr[:,:,3]=(a*keep).astype(np.uint8); b=Image.fromarray(arr)
b = b.crop(b.getchannel("A").getbbox())
cnts,_ = cv2.findContours((np.array(b)[:,:,3]>20).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
r = cv2.minAreaRect(max(cnts,key=cv2.contourArea)); ang=r[2]
if r[1][0]>r[1][1]: ang-=90
if 1<abs(ang)<20: b=b.rotate(ang,expand=True,resample=Image.BICUBIC); b=b.crop(b.getchannel("A").getbbox())
rgb = ImageEnhance.Contrast(b.convert("RGB")).enhance(1.05)
b = Image.merge("RGBA",[*rgb.split(), b.getchannel("A")])
W,H=800,1000; bg=Image.new("RGB",(W,H),"white")
th=int(H*0.86); s=th/b.height; b=b.resize((max(1,int(b.width*s)),th),Image.LANCZOS)
px=(W-b.width)//2; py=int(H*0.06)
sh=Image.new("L",(W,H),0); d=ImageDraw.Draw(sh); base=py+b.height
d.ellipse((W//2-int(b.width*0.75),base-14,W//2+int(b.width*0.75),base+14),fill=70)
bg.paste(Image.new("RGB",(W,H),(120,110,100)),(0,0),sh.filter(ImageFilter.GaussianBlur(12)))
bg.paste(b,(px,py),b); bg.save(sys.argv[2],"JPEG",quality=88,optimize=True); print("ok")
