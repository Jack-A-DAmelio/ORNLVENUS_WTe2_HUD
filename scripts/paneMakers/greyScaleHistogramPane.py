import os,re,json,csv,tifffile,numpy as np,matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.widgets import RectangleSelector
from pathlib import Path
from PIL import Image

# ==========================================================
# IO
# ==========================================================

def load_image(p): return tifffile.imread(p)

def extract_run_number(f):
    m=re.search(r"_(\d+_\d+)",f)
    if m: return m.group(1)
    raise ValueError("no run number found")

# ==========================================================
# ROI
# ==========================================================

def select_roi(img):
    roi={}
    fig,ax=plt.subplots(figsize=(8,8))
    ax.imshow(img,cmap="gray");ax.set_title("Select ROI then close")

    def on_select(e1,e2):
        x1,y1,x2,y2=int(e1.xdata),int(e1.ydata),int(e2.xdata),int(e2.ydata)
        roi["x1"],roi["x2"]=min(x1,x2),max(x1,x2)
        roi["y1"],roi["y2"]=min(y1,y2),max(y1,y2)
        print("ROI:",roi)

    RectangleSelector(ax,on_select,useblit=True,button=[1],interactive=True)
    plt.show()

    if not roi: raise RuntimeError("No ROI selected")
    return roi

def save_roi(roi,out):
    p=Path(out)/"ROI_coordinates.json"
    json.dump(roi,open(p,"w"),indent=2)
    print("saved:",p)

# ==========================================================
# IMAGE OPS
# ==========================================================

def crop_image(img,roi):
    return img[roi["y1"]:roi["y2"],roi["x1"]:roi["x2"]]

def calculate_histogram(img,bins=500):
    px=img.ravel()
    px=px[(px!=0.0)&(px!=1.0)]
    h,e=np.histogram(px,bins=bins,range=(px.min(),px.max()))
    g=(e[:-1]+e[1:])/2
    return g,h

# ==========================================================
# MODELS
# ==========================================================

def gaussian(x,a,m,s):
    return a*np.exp(-(x-m)**2/(2*s**2))

def double_gaussian(x,a1,m1,s1,a2,m2,s2):
    return gaussian(x,a1,m1,s1)+gaussian(x,a2,m2,s2)

def fit_gaussians(g,h):
    mu=np.average(g,weights=h)
    sg=np.sqrt(np.average((g-mu)**2,weights=h))
    amp=h.max()

    p1,_=curve_fit(gaussian,g,h,p0=[amp,mu,sg])

    mid=len(g)//2
    p2,_=curve_fit(
        double_gaussian,g,h,
        p0=[amp/2,g[mid//2],sg/2,amp/2,g[mid+mid//2],sg/2],
        maxfev=10000
    )

    r1=np.sum((h-gaussian(g,*p1))**2)
    r2=np.sum((h-double_gaussian(g,*p2))**2)

    return p1,p2,r1,r2

# ==========================================================
# PLOTTING
# ==========================================================

def plot_gaussian_comparison(g,h,p1,p2,r1,r2,out,run,save=True,show=False):
    plt.figure(figsize=(8,6))
    plt.plot(g,h,label="hist")
    plt.plot(g,gaussian(g,*p1),label=f"1G RSS={r1:.2e}")
    plt.plot(g,double_gaussian(g,*p2),label=f"2G RSS={r2:.2e}")
    plt.xlabel("grey");plt.ylabel("count");plt.title(run)
    plt.legend();plt.tight_layout()

    if save:
        p=Path(out)/f"GaussianFitComparison_{run}.png"
        plt.savefig(p,dpi=300,bbox_inches="tight")
        print("saved:",p)

    if show:
        plt.show()
    else:
        plt.close()

# ==========================================================
# CSV SYSTEM (FIXED: CSV WRITER USED PROPERLY)
# ==========================================================

def init_dataset_csv(folder):
    csv_path=Path(folder)/"parameters.csv"
    csv_path.parent.mkdir(parents=True,exist_ok=True)

    if not csv_path.exists():
        with open(csv_path,"w",newline="") as f:
            csv.writer(f).writerow([
                "tiff","run","roi",
                "g1_amp","g1_mean","g1_sigma","rss_1g",
                "g2_amp1","g2_mean1","g2_sigma1",
                "g2_amp2","g2_mean2","g2_sigma2","rss_2g",
                "delta_rss","better_model","error"
            ])

    return csv_path

def append_dataset_row(csv_path,tiff,run,roi,p1,p2,r1,r2,error=None):

    g1_amp,g1_mean,g1_sigma=p1
    g2_amp1,g2_mean1,g2_sigma1,g2_amp2,g2_mean2,g2_sigma2=p2

    delta_rss=r1-r2
    better_model="1G" if r1<r2 else "2G"

    row=[
        str(tiff),
        run,
        json.dumps(roi,separators=(",",":")),
        g1_amp,g1_mean,g1_sigma,r1,
        g2_amp1,g2_mean1,g2_sigma1,
        g2_amp2,g2_mean2,g2_sigma2,
        r2,
        delta_rss,
        better_model,
        error if error else ""
    ]

    with open(csv_path,"a",newline="") as f:
        csv.writer(f).writerow(row)

# ==========================================================
# SINGLE IMAGE PIPELINE
# ==========================================================

def process_image(path,out,bins,show=False):
    img=load_image(path)
    run=extract_run_number(Path(path).name)

    roi=select_roi(img)
    save_roi(roi,out)

    crop=crop_image(img,roi)
    g,h=calculate_histogram(crop,bins)

    error=None
    try:
        p1,p2,r1,r2=fit_gaussians(g,h)
    except Exception as e:
        p1=[np.nan]*3
        p2=[np.nan]*6
        r1=r2=np.nan
        error=str(e)

    plot_gaussian_comparison(g,h,p1,p2,r1,r2,out,run,show=show)

    csv_path=init_dataset_csv(out)
    append_dataset_row(csv_path,path,run,roi,p1,p2,r1,r2,error)

    return {"run":run,"roi":roi,"p1":p1,"p2":p2,"r1":r1,"r2":r2,"error":error}

# ==========================================================
# BATCH PIPELINE
# ==========================================================

def create_roi_panes(folder,dest,roi,bins=100,show=False):
    folder,dest=Path(folder),Path(dest)
    dest.mkdir(parents=True,exist_ok=True)

    csv_path=init_dataset_csv(dest)

    files=sorted(folder.glob("*.tif"))
    if not files:
        raise ValueError("no tif files found")

    print(f"[INFO] {len(files)} files")
    pane=[]

    for i,f in enumerate(files):
        print(f"[PROCESS] {i+1}/{len(files)} {f.name}")

        img=np.array(Image.open(f))
        x1,x2,y1,y2=roi
        crop=img[y1:y2,x1:x2]

        g,h=calculate_histogram(crop,bins)

        error=None
        try:
            p1,p2,r1,r2=fit_gaussians(g,h)
        except Exception as e:
            p1=[np.nan]*3
            p2=[np.nan]*6
            r1=r2=np.nan
            error=str(e)

        run=f.stem

        plot_gaussian_comparison(g,h,p1,p2,r1,r2,dest,run,save=True,show=show)

        append_dataset_row(csv_path,f,run,roi,p1,p2,r1,r2,error)

        pane.append(Path(dest)/f"GaussianFitComparison_{run}.png")

    return pane