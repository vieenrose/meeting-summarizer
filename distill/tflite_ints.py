"""Minimal reader of a .tflite flatbuffer: the int2/int4/int8 weight tensors, their byte span in the file,
and (un)packing of the sub-byte integers (used by distill/inject_litertlm.py)."""
import numpy as np
from ai_edge_litert import schema_py_generated as fb
TT={v:k for k,v in fb.TensorType.__dict__.items() if not k.startswith("_")}
class TFL:
    def __init__(self, path):
        self.path=path; self.buf=open(path,"rb").read(); self.m=fb.Model.GetRootAsModel(self.buf,0)
    def tensors(self):
        seen=set()
        for s in range(self.m.SubgraphsLength()):
            sg=self.m.Subgraphs(s)
            for i in range(sg.TensorsLength()):
                t=sg.Tensors(i)
                if TT.get(t.Type()) in ("INT2","INT4","INT8") and t.Buffer() not in seen:
                    bb=self.m.Buffers(t.Buffer())
                    if (bb.Size() or bb.DataLength())>100000:
                        seen.add(t.Buffer()); yield t
    def span(self, t):
        bb=self.m.Buffers(t.Buffer())
        if bb.Offset():
            return bb.Offset(), bb.Size()
        o=bb._tab.Offset(4)
        start=bb._tab.Vector(o); return start, bb._tab.VectorLen(o)
    def raw(self, t):
        a,n=self.span(t); return np.frombuffer(self.buf, dtype=np.uint8, count=n, offset=a)
def unpack(a,typ):
    a=a.astype(np.uint8)
    if typ=="INT4":
        lo=(a&0xF).astype(np.int8); hi=(a>>4).astype(np.int8)
        return np.stack([np.where(lo>7,lo-16,lo),np.where(hi>7,hi-16,hi)],-1).reshape(-1)
    if typ=="INT2":
        v=[((a>>s)&3).astype(np.int8) for s in (0,2,4,6)]
        return np.stack([np.where(x>1,x-4,x) for x in v],-1).reshape(-1)
    return a.view(np.int8)
def pack(q,typ):
    q=q.reshape(-1).astype(np.int16)
    if typ=="INT4":
        u=(q&0xF).astype(np.uint8); return (u[0::2]|(u[1::2]<<4)).astype(np.uint8)
    if typ=="INT2":
        u=(q&0x3).astype(np.uint8); return (u[0::4]|(u[1::4]<<2)|(u[2::4]<<4)|(u[3::4]<<6)).astype(np.uint8)
    return q.astype(np.int8).view(np.uint8)
