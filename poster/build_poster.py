# Gera o pôster do Congresso Unificado da UFLA 2026 a partir do template e das figuras do relatório.
# Uso: python poster/build_poster.py poster/template_congresso_ufla.pptx poster/poster.pptx figs/relatorio
import sys, copy
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree
from PIL import Image

SRC, OUT, FIGS = sys.argv[1], sys.argv[2], sys.argv[3]
NAVY = RGBColor(0x00, 0x20, 0x60)
CYAN = RGBColor(0x00, 0x9F, 0xE3)
INK = RGBColor(0x1A, 0x1A, 0x2E)
MUTED = RGBColor(0x4A, 0x55, 0x68)
TINT = RGBColor(0xEA, 0xF2, 0xFB)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Calibri"

BODY, CAP, HEAD = 31, 24, 48

prs = Presentation(SRC)
s = prs.slides[0]
# keep only the congress banner (everything above y = 7.2")
for sh in list(s.shapes):
    if sh.top > Inches(7.2) or sh.name == "AutoShape 20":
        sh._element.getparent().remove(sh._element)

def runs(p, parts, size, color=INK, bold=False):
    """parts: str or list of (text, dict)"""
    if isinstance(parts, str):
        parts = [(parts, {})]
    out = []
    for t, o in parts:
        pieces = t.split("α_L")
        for k, piece in enumerate(pieces):
            if piece: out.append((piece, o))
            if k < len(pieces) - 1:
                out.append(("α", o)); out.append(("L", dict(o, sub=True)))
    parts = out
    for t, o in parts:
        r = p.add_run()
        r.text = t
        f = r.font
        f.name = FONT
        f.size = Pt(o.get("size", size))
        f.bold = o.get("bold", bold)
        f.italic = o.get("italic", False)
        f.color.rgb = o.get("color", color)
        if o.get("sub"):
            r._r.get_or_add_rPr().set("baseline", "-25000")

def textbox(x, y, w, h, paras, size=BODY, color=INK, bold=False, align=PP_ALIGN.LEFT,
            bullet=False, space_after=10, anchor=MSO_ANCHOR.TOP, name=None, line=None):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name: tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        if line: p.line_spacing = line
        if bullet:
            pPr = p._p.get_or_add_pPr()
            pPr.set("marL", str(int(Inches(0.45))))
            pPr.set("indent", str(-int(Inches(0.45))))
            bc = etree.SubElement(pPr, qn("a:buClr"))
            etree.SubElement(bc, qn("a:srgbClr")).set("val", "009FE3")
            etree.SubElement(pPr, qn("a:buFont")).set("typeface", "Arial")
            etree.SubElement(pPr, qn("a:buChar")).set("char", "•")
        runs(p, para, size, color, bold)
    return tb

def box(x, y, w, h, fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12, name=None):
    sh = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if name: sh.name = name
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    sh.line.fill.background()
    sh.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = radius
    return sh

def header(x, y, w, text, h=0.95):
    sh = box(x, y, w, h, NAVY, radius=0.5, name="Cabeçalho " + text)
    tf = sh.text_frame
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    runs(p, text, HEAD, WHITE, bold=True)
    return y + h

def badge(x, y, n, d=0.8):
    c = box(x, y, d, d, CYAN, shape=MSO_SHAPE.OVAL, name=f"Marcador {n}")
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    runs(p, str(n), 36, WHITE, bold=True)

def picture(name, x, y, w=None, h=None, alt=""):
    path = f"{FIGS}/{name}.png"
    iw, ih = Image.open(path).size
    if w is None: w = h * iw / ih
    if h is None: h = w * ih / iw
    pic = s.shapes.add_picture(path, Inches(x), Inches(y), Inches(w), Inches(h))
    pic.name = name
    pic._element.nvPicPr.cNvPr.set("descr", alt)
    return pic, y + h

# ---------- title block ----------
PW = prs.slide_width / 914400
M = 0.7
title = box(M, 7.55, PW - 2 * M, 2.15, NAVY, radius=0.15, name="Título")
tf = title.text_frame; tf.word_wrap = True
tf.margin_left = tf.margin_right = Inches(0.6)
tf.vertical_anchor = MSO_ANCHOR.MIDDLE
p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
runs(p, "Aplicação de Métodos de Aprendizado de Máquina na Análise de Simulações de Dinâmica Molecular", 64, WHITE, bold=True)

textbox(M, 9.95, PW - 2 * M, 0.6,
        [[("Pedro Henrique Ázara de Almeida¹, Igor Saulo Santos de Oliveira²", {})]],
        size=40, bold=True, color=NAVY, align=PP_ALIGN.CENTER, name="Autores")
textbox(M, 10.65, PW - 2 * M, 0.5,
        ["¹ Apresentador(a), Bolsista do Programa PIBIC/CNPq, Departamento de Física, Universidade Federal de Lavras;  "
         "² Departamento de Física, Universidade Federal de Lavras"],
        size=24, color=MUTED, align=PP_ALIGN.CENTER, name="Afiliações")

# ---------- columns ----------
TOP = 11.55
LX, LW = M, 15.0
RX = LX + LW + 0.8
RW = PW - M - RX
GAP = 0.45

# left column
y = header(LX, TOP, LW, "INTRODUÇÃO") + GAP
textbox(LX + 0.2, y, LW - 0.2, 3.6, [
    "A dinâmica molecular gera milhares de conformações; os estados metaestáveis precisam ser extraídos dos dados.",
    "O agrupamento não supervisionado (K-means) de descritores estruturais é a abordagem usual.",
    [("Os ângulos diedros (φ, ψ) são os melhores descritores, mas são ", {}), ("periódicos", {"bold": True}),
     (": −179° e 179° estão a só 2° de distância.", {})],
], bullet=True, space_after=12, name="Texto Introdução")
y += 3.6 + 0.45

y = header(LX, y, LW, "OBJETIVOS") + GAP
textbox(LX + 0.2, y, LW - 0.2, 2.2, [
    "Identificar os estados metaestáveis do hexâmero de alanina por agrupamento.",
    "Desenvolver um K-means para variáveis periódicas com convergência garantida.",
    "Verificar se o critério geométrico basta ou se é preciso um critério cinético.",
], bullet=True, space_after=12, name="Texto Objetivos")
y += 2.2 + 0.45

y = header(LX, y, LW, "MATERIAL E MÉTODOS") + GAP
sw = 8.2
_, yb = picture("S_estrutura", LX, y, w=sw,
                alt="Estrutura do Ace-(Ala)6-NH2 com a ligação cis entre Ala3 e Ala4 em destaque")
textbox(LX + sw + 0.35, y + 0.1, LW - sw - 0.35, yb - y, [
    "Ace-(Ala)₆-NH₂, átomo unido (42 sítios).",
    "20 ns a 500 K e 10 ns a 300 K; 1 frame a cada 10 ps.",
    "Descritores: Rg, RMSD e 12 diedros (φ, ψ).",
], size=28, bullet=True, space_after=10, name="Texto sistema")
textbox(LX, yb + 0.08, sw, 0.5, ["Ligação Ala3–Ala4 cis em todos os frames."],
        size=CAP, color=MUTED, align=PP_ALIGN.CENTER, name="Legenda estrutura")
y = yb + 0.75

# circular mean figure beside the key argument
fh = 4.9
pic, _ = picture("B_media_circular", LX, y, h=fh,
                 alt="Quatro ângulos perto de ±180 graus: a média aritmética dá 0 grau, a média circular dá 180 graus")
fw = pic.width / 914400
textbox(LX + fw + 0.4, y + 0.2, LW - fw - 0.4, fh, [
    [("O K-means só converge se a atualização ", {}), ("minimizar a mesma distância", {"bold": True}),
     (" usada na atribuição.", {})],
    [("Perto de ±180°, a média aritmética cai do lado oposto do círculo. Isso atinge ", {}),
     ("11,6%", {"bold": True, "color": NAVY}), (" dos ângulos a 500 K.", {})],
], size=29, space_after=14, name="Texto argumento")
y += fh + 0.35

# variants table
rows = [("variante", "atribuição", "atualização", "coerente"),
        ("tutorial", "imagem mínima", "média aritmética", "não"),
        ("mista", "imagem mínima", "média circular", "parcial"),
        ("corda", "1 − cos Δθ", "média circular", "sim")]
th = 0.56
tbl = s.shapes.add_table(len(rows), 4, Inches(LX), Inches(y), Inches(LW), Inches(th * len(rows)))
tbl.name = "Tabela variantes"
t = tbl.table
for j, wv in enumerate([3.0, 4.1, 4.9, 3.0]):
    t.columns[j].width = Inches(wv)
for i, row in enumerate(rows):
    t.rows[i].height = Inches(th)
    for j, val in enumerate(row):
        c = t.cell(i, j)
        c.fill.solid()
        c.fill.fore_color.rgb = NAVY if i == 0 else (TINT if i == 3 else WHITE)
        c.margin_left = c.margin_right = Inches(0.15)
        c.margin_top = c.margin_bottom = Inches(0.04)
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = c.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        runs(p, val, 26, WHITE if i == 0 else INK, bold=(i == 0 or i == 3))
tblPr = t._tbl.tblPr
for k in ("bandRow", "firstRow"):
    tblPr.set(k, "0")
y += th * len(rows) + 0.45
textbox(LX + 0.2, y, LW - 0.2, 1.1, [
    "Critério cinético: escalas de tempo implícitas de modelos de Markov (20 e 40 microestados).",
], bullet=True, name="Texto cinética")
LEFT_END = y + 1.1

# right column
y = header(RX, TOP, RW, "RESULTADOS E DISCUSSÃO") + GAP + 0.05

def finding(y, n, head, text):
    badge(RX, y, n)
    textbox(RX + 1.05, y + 0.02, RW - 1.05, 0.8, [head], size=38, bold=True, color=NAVY,
            anchor=MSO_ANCHOR.MIDDLE, name=f"Achado {n}")
    textbox(RX + 1.05, y + 0.95, RW - 1.05, 1.0, [text], size=29, name=f"Texto achado {n}", space_after=0)

finding(y, 1, "Só a variante coerente converge sempre",
        [("Em 50 inicializações por K, ", {}), ("tutorial", {"italic": True}),
         (" fica presa em ciclo limite em até 58% delas; ", {}), ("corda", {"italic": True}),
         (" nunca, e seu objetivo não sobe em nenhuma das 550 rodadas.", {})])
y += 2.15
_, y = picture("B_convergencia", RX + (RW - 13.2) / 2, y, w=13.2,
               alt="Fração de inicializações presas em ciclo limite e com objetivo subindo, por K, para as três variantes")
y += 0.55

finding(y, 2, "K = 4: quatro famílias de conformações",
        [("A partição de referência dá populações de 29%, 29%, 28% e 14%. Mas 50 inicializações chegam a ", {}),
         ("43 mínimos locais distintos", {"bold": True}), (".", {})])
y += 2.15
mw = 8.4
pic, yb = picture("E_medoides", RX, y, w=mw,
                  alt="Estrutura representativa de cada um dos quatro estados")
textbox(RX + mw + 0.4, y + 0.3, RW - mw - 0.4, yb - y - 0.3, [
    "Nome do estado: a região de Ramachandran de Ala1 a Ala6 — α, β, P (PPII), L (α_L).",
    [("O estado 2, único com α_L, é o mais persistente: ", {}), ("88%", {"bold": True, "color": NAVY}),
     (" de chance de continuar nele após 10 ps.", {})],
], size=29, bullet=True, space_after=14, name="Texto estados")
y = yb + 0.55

finding(y, 3, "O critério cinético revela um processo lento",
        [("Os 4 estados geométricos não são markovianos (a). Com 40 microestados surge um processo separado, ", {}),
         ("t₁\u00a0=\u00a00,34\u00a0ns", {"bold": True, "color": NAVY}),
         (" (IC 95%: 0,24–0,57 ns): a entrada e saída de α_L.", {})])
y += 2.15
_, y = picture("C_its", RX + (RW - 14.0) / 2, y, w=14.0,
               alt="Escalas de tempo implícitas em função do tempo de atraso para 4 estados, 20 e 40 microestados")
RIGHT_END = y
print("left ends", round(LEFT_END, 2), "right ends", round(RIGHT_END, 2))

# ---------- conclusions ----------
y = max(LEFT_END, RIGHT_END) + 0.5
CY = y
header(M, y, PW - 2 * M, "CONCLUSÕES")
y += 0.95 + GAP
cards = [
    [("Para ângulos, a atualização deve ser a ", {}), ("média circular", {"bold": True}),
     (": só o par coerente garante convergência.", {})],
    [("A geometria ", {}), ("não fixa", {"bold": True}),
     (" o número de estados: cotovelo fraco e muitos mínimos locais.", {})],
    [("A cinética fixa: um processo lento de ", {}), ("≈\u00a00,34\u00a0ns", {"bold": True}),
     (", a entrada e saída de α_L.", {})],
]
cw = (PW - 2 * M - 2 * 0.5) / 3
ch = 2.5
for k, txt in enumerate(cards):
    cx = M + k * (cw + 0.5)
    box(cx, y, cw, ch, TINT, radius=0.1, name=f"Cartão conclusão {k+1}")
    badge(cx + 0.35, y + (ch - 0.8) / 2, k + 1)
    textbox(cx + 1.4, y + 0.2, cw - 1.75, ch - 0.4, [txt], size=29, anchor=MSO_ANCHOR.MIDDLE,
            name=f"Texto conclusão {k+1}", space_after=0)
y += ch + 0.45
CONC_END = y

# ---------- footer: references + acknowledgements ----------
FW = 22.0
header(M, y, FW, "Referências", h=0.8)
header(M + FW + 0.6, y, PW - 2 * M - FW - 0.6, "Agradecimentos", h=0.8)
y += 0.8 + 0.25
refs = [
    "Lab Course MolSim 2025, cap. 7: Machine Learning for Molecular Simulation. Universidade de Amsterdã.",
    "FRENKEL, D.; SMIT, B. Understanding Molecular Simulation. 3. ed. Academic Press, 2023.",
    "LLOYD, S. P. Least squares quantization in PCM. IEEE Trans. Inf. Theory, v. 28, p. 129, 1982.",
    "PRINZ, J.-H. et al. Markov models of molecular kinetics. J. Chem. Phys., v. 134, 174105, 2011.",
]
textbox(M + 0.2, y, FW - 0.2, 2.2, refs, size=23, color=MUTED, bullet=True, space_after=4, name="Referências")
textbox(M + FW + 0.8, y, PW - 2 * M - FW - 0.8, 2.6,
        ["Ao CNPq, pela bolsa de Iniciação Científica (PIBIC), e ao Departamento de Física da UFLA."],
        size=26, color=MUTED, name="Texto agradecimentos")
print("footer text starts", round(y, 2), "slide h", round(prs.slide_height / 914400, 2))
prs.save(OUT)
