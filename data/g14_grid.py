"""Audio and visual speech codecs for a complete GRID utterance; no word timing is an inference input."""
import numpy as np
COLORS=("red","blue","green","white")
LETTERS=tuple("abcdefghijklmnopqrstuvxyz")
DIGITS=("zero","one","two","three","four","five","six","seven","eight","nine")
SLOTS=(COLORS,LETTERS,DIGITS)
T=96

def acoustic(pcm):
    sr=16000;pcm=np.asarray(pcm,dtype=np.float32)
    if pcm.dtype.kind=="f" and np.max(np.abs(pcm),initial=0)>1.5:pcm=pcm/32768
    arr=np.zeros(T*512+512,np.float32);n=min(len(arr),len(pcm));arr[:n]=pcm[:n]
    spec=np.abs(np.fft.rfft(arr[np.arange(T)[:,None]*512+np.arange(512)]*np.hanning(512),axis=1))**2
    hz=np.fft.rfftfreq(512,1/sr);m=np.linspace(2595*np.log10(1+55/700),2595*np.log10(1+7600/700),66)
    edges=700*(10**(m/2595)-1);a,b,c=edges[:-2],edges[1:-1],edges[2:]
    fb=np.maximum(0,np.minimum((hz[:,None]-a)/(b-a),(c-hz[:,None])/(c-b)))
    return np.clip((np.log10(np.maximum(spec@fb,1e-7))+5.5)/7,0,1).astype("f2"),min(T,int(np.ceil(n/512)))

def mouth_field(mouth):
    # A 12 x 8 luminance mouth image at the original 25 Hz, sampled onto the acoustic clock.
    pix=np.asarray(mouth)
    if pix.shape[1:]!=(64,96):raise ValueError("Unexpected mouth crop")
    tiny=pix.reshape(len(pix),8,8,12,8).mean((2,4)).reshape(len(pix),96)
    idx=np.minimum(((np.arange(T)*.032+.016)*25).astype(int),len(pix)-1)
    return np.rint(tiny[idx]).astype("u1")

def labels(text):
    words=text.lower().split()
    if len(words)!=6:raise ValueError("Incomplete GRID utterance")
    return np.array([v.index(words[h]) for v,h in zip(SLOTS,(1,3,4))],np.int32)
