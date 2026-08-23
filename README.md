# ระบบเตือนภัยล่วงหน้าและอธิบายความเสี่ยงเครดิตตราสารหนี้ไทยด้วย AI (Project LIME)
### Explainable AI (LIME & SHAP) Early Warning System for Thai Corporate Bond Defaults

> ระบบปัญญาประดิษฐ์อธิบายผลความเสี่ยงเครดิต (XAI) สำหรับการประเมินโอกาสผิดนัดชำระหนี้และการปรับโครงสร้างหนี้ของบริษัทในตลาดทุนไทย ครอบคลุม 30 ปัจจัยเสี่ยงทางการเงิน สภาพคล่อง มหภาค และ ESG พร้อมสถาปัตยกรรมเชื่อมต่อข้อมูล 2 ชุด (DataAdapter) และกลไกตรวจจับความเสี่ยง 3 ชั้น (วิธี A, B, C)

---

## 📑 สารบัญ (Table of Contents)
1. [ภาพรวมของโครงการ (Overview)](#-1-ภาพรวมของโครงการ-overview)
2. [สถาปัตยกรรมระบบและจุดเด่นหลัก (Key Features)](#-2-สถาปัตยกรรมระบบและจุดเด่นหลัก-key-features)
3. [โครงสร้างไฟล์และโฟลเดอร์ในโครงการ (Project Structure)](#-3-โครงสร้างไฟล์และโฟลเดอร์ในโครงการ-project-structure)
4. [โครงสร้างฐานข้อมูล SQLite (`lime_credit.db`)](#-4-โครงสร้างฐานข้อมูล-sqlite-lime_creditdb)
5. [การติดตั้งและข้อกำหนดของระบบ (Installation & Setup)](#-5-การติดตั้งและข้อกำหนดของระบบ-installation--setup)
6. [คู่มือการใช้งานคำสั่ง (Command-Line Reference)](#-6-คู่มือการใช้งานคำสั่ง-command-line-reference)
7. [รายการ 30 ปัจจัยเสี่ยงในการประเมิน (30 Determinants)](#-7-รายการ-30-ปัจจัยเสี่ยงในการประเมิน-30-determinants)
8. [ผลการประเมินความแม่นยำวิธี A, B, C (Accuracy Benchmark)](#-8-ผลการประเมินความแม่นยำวิธี-a-b-c-accuracy-benchmark)
9. [เอกสารคู่มือฉบับเต็ม PDF (Documentation)](#-9-เอกสารคู่มือฉบับเต็ม-pdf-documentation)

---

## 🎯 1. ภาพรวมของโครงการ (Overview)

โครงการ **iBond LIME** ถูกพัฒนาขึ้นเพื่อยกระดับความโปร่งใสและความแม่นยำของระบบเตือนภัยล่วงหน้า (Early Warning System) ในตลาดตราสารหนี้และตลาดทุนไทย โดยแก้ปัญหา "กล่องดำ" (Black-Box Problem) ของโมเดล Machine Learning แบบดั้งเดิม ด้วยการผสานเทคโนโลยี **LIME (Local Interpretable Model-agnostic Explanations)** และ **SHAP (SHapley Additive exPlanations)** เข้ากับแบบจำลองประเมินความเสี่ยงเครดิต

ระบบสามารถแจกแจงได้ว่า ทำไมบริษัทหนึ่งจึงมีความน่าจะเป็นในการผิดนัดชำระหนี้ (Probability of Default: PD) สูงหรือต่ำ ปัจจัยใดที่ผลักดันความเสี่ยง และปัจจัยใดที่ช่วยพยุงสถานะของบริษัท พร้อมทั้งแสดงผลการวิเคราะห์เป็นภาพกราฟิก 3 แผง (Three-Panel Visualization) ในรูปแบบไฟล์ `.jpg` และ `.png` ความละเอียดสูง

---

## ⚡ 2. สถาปัตยกรรมระบบและจุดเด่นหลัก (Key Features)

1. **ระบบเชื่อมต่อข้อมูล 2 ชุดอัตโนมัติ (Dual-Dataset DataAdapter)**:
   - **ชุดข้อมูลเดิม (Dataset 1)**: `ibond_33features_panel` (16,986 แถว, 219–293 ผู้ออกตราสารหนี้)
   - **ชุดข้อมูลใหม่ (Dataset 2)**: `ibond_33features_panel_941firm` (187,007 แถว, 941 บริษัททั้ง SET และ mai ตั้งแต่ปี 2007 ถึงสิงหาคม 2026)
   - มีเมนูแบบ Interactive ให้เลือกหน้าจอ หรือส่งพารามิเตอร์ผ่านคำสั่ง Command-Line ได้โดยตรง
2. **การอธิบายผลแบบประกบคู่ (Repeated LIME + Exact Tree SHAP)**:
   - **Repeated-Seed LIME**: สุ่มรบกวน 5,000 จุด ทำซ้ำ 8 Seed พร้อมคำนวณแถบความเชื่อมั่น (Confidence Interval) เพื่อยืนยันความเสถียรของผลลัพธ์
   - **Exact Additive Tree SHAP**: คำนวณค่าน้ำหนักเสริมความเสี่ยงจริงตามทฤษฎีเกม (Game Theory) ซึ่งผลรวมของค่าน้ำหนักตรงกับค่า PD รวมของโมเดลอย่างสมบูรณ์
   - **Distance from Median**: แสดงระยะห่างของตัวแปรเทียบกับค่ามัธยฐานของตลาดในหน่วยส่วนเบี่ยงเบนมาตรฐาน (SD)
3. **ฐานข้อมูลขนาดกะทัดรัดพร้อมระบบ Auto-Unzip**:
   - บรรจุฐานข้อมูลเฉพาะตารางที่ใช้งานจริงในไฟล์ `lime_credit.db.zip` (ขนาดเพียง 23.8 MB) เมื่อรันโปรแกรมครั้งแรก ระบบจะทำการคลายซิปเป็น `lime_credit.db` ให้อัตโนมัติในเวลา 1 วินาที
4. **กลไกการคัดกรองความเสี่ยง 3 ชั้น (Three-Layer Review Framework)**:
   - ตรวจจับความเสี่ยงผ่าน **วิธี A** (เกณฑ์ระดับความเสี่ยง), **วิธี B** (เกณฑ์ความเปราะบางต่อ Shock), และ **วิธี C** (เกณฑ์ความเสี่ยงแฝงทางงบดุล)

---

## 📂 3. โครงสร้างไฟล์และโฟลเดอร์ในโครงการ (Project Structure)

```
iBond_LIME/
├── lime33_adapter_panel.py     # โปรแกรมหลัก (DataAdapter + เมนูเลือกชุดข้อมูล + CLI + ส่งออกภาพ JPG)
├── lime_feature33.py           # โปรแกรมเวอร์ชันดั้งเดิม
├── data_adapter.py             # โมดูล DataAdapter เชื่อมต่อ SQLite พร้อมระบบ Auto-Unzip
├── evaluate_methods_abc.py     # สคริปต์ประเมินความแม่นยำวิธี A, B, C เปรียบเทียบทั้ง 2 ชุดข้อมูล
├── batch_run_20_issuers.py     # สคริปต์ประมวลผลและสร้างรูปภาพของ 20 บริษัทตัวอย่าง
├── a_approach.py               # โมดูลประเมินความเสี่ยง 3 ชั้น (Three-Layer Review Framework)
├── firm_shock_panel.py         # โมดูลวิเคราะห์ความเปราะบางของปัจจัย (Shock Fragility Ladder)
├── cmdf_tree_classify.py       # โมดูลสร้างโมเดล Machine Learning (CatBoost OOF Classifier)
├── lime_credit.db.zip          # ไฟล์ฐานข้อมูล SQLite บีบอัด (23.8 MB คลายซิปอัตโนมัติ)
├── lime33.tex                  # ซอร์สโค้ดคู่มือ LaTeX ภาษาไทยฉบับสมบูรณ์ (50 หน้า)
├── lime33.pdf                  # คู่มือการใช้งานฉบับสมบูรณ์ PDF (50 หน้า, ความละเอียดสูง)
├── requirements.txt            # รายการไลบรารี Python ที่ต้องติดตั้ง
├── .gitignore                  # รายการไฟล์ที่ไม่ต้องการนำขึ้น Git
├── README.md                   # เอกสารคู่มือโครงการภาษาไทย
└── tex_out/
    ├── lime_jpg/               # โฟลเดอร์เก็บภาพผลการอธิบายความเสี่ยง 3 แผง (.jpg)
    ├── lime_figs/              # โฟลเดอร์เก็บภาพผลการวิเคราะห์ (.png)
    └── summary_941_firms.csv   # สรุปรายชื่อ 941 บริษัท, ช่วงเวลา, และ % ความครบถ้วนของฟีเจอร์
```

---

## 🗄️ 4. โครงสร้างฐานข้อมูล SQLite (`lime_credit.db`)

ฐานข้อมูล `lime_credit.db` ประกอบด้วย 5 ตารางหลักที่ผ่านการสร้างดัชนี (Indexes) อย่างมีประสิทธิภาพ:

| ชื่อตาราง (Table Name) | จำนวนแถว | จำนวนคอลัมน์ | รายละเอียดข้อมูล |
| :--- | :---: | :---: | :--- |
| **`ibond_33features_panel`** | 16,986 แถว | 61 คอลัมน์ | ชุดข้อมูลเดิม: ผู้ออกตราสารหนี้ 219–293 บริษัท (ม.ค. 2007 – ม.ค. 2026) |
| **`ibond_33features_panel_941firm`** | 187,007 แถว | 95 คอลัมน์ | ชุดข้อมูลใหม่: 941 บริษัททั้ง SET/mai ครอบคลุมถึง ส.ค. 2026 |
| **`ibond_default_payment`** | 50 แถว | 12 คอลัมน์ | ทะเบียนประวัติการผิดนัดชำระหนี้ตราสารหนี้ทางการของ ThaiBMA |
| **`firm_issuer_mapping`** | 985 แถว | 6 คอลัมน์ | ตารางจับคู่ Ticker Symbol กับรหัสผู้ออกตราสารหนี้ |
| **`ibond_issuer`** | 678 แถว | 8 คอลัมน์ | ข้อมูลบริษัทและหมวดธุรกิจของผู้ออกตราสารหนี้ |

---

## 💻 5. การติดตั้งและข้อกำหนดของระบบ (Installation & Setup)

### ข้อกำหนดพื้นฐาน:
* Python 3.9 ขึ้นไป
* ระบบปฏิบัติการ: Windows, macOS, หรือ Linux

### ขั้นตอนการติดตั้ง:
```bash
# 1. Clone Repository มายังเครื่องของคุณ
git clone https://github.com/kkabinsky/iBond_LIME.git
cd iBond_LIME

# 2. ติดตั้งแพ็กเกจ Python ทั้งหมด
pip install -r requirements.txt
```

---

## 🚀 6. คู่มือการใช้งานคำสั่ง (Command-Line Reference)

### 6.1 รันแบบเมนูเลือกหน้าจอ (Interactive Menu Mode)
เหมาะสำหรับการใช้งานทั่วไป สามารถเลือกชุดข้อมูลและพิมพ์ชื่อย่อบริษัทได้ทันที:
```bash
python lime33_adapter_panel.py
```
*หน้าจอจะแสดงตัวเลือก [1] ชุดข้อมูลเดิม หรือ [2] ชุดข้อมูลใหม่ 941 บริษัท*

### 6.2 รันระบุพารามิเตอร์โดยตรง (Direct CLI Mode)
* **วิเคราะห์รายบริษัทบนชุดข้อมูลใหม่ 941 บริษัท (Dataset 2):**
  ```bash
  python lime33_adapter_panel.py -d 2 --issuer PTT
  python lime33_adapter_panel.py -d 2 --issuer A
  python lime33_adapter_panel.py -d 2 --issuer PRIME
  ```

* **วิเคราะห์ทุกบริษัทที่อยู่ในกลุ่มเสี่ยงสูง (High-Risk Issuers):**
  ```bash
  python lime33_adapter_panel.py -d 2 --all-high-risk
  ```

* **วิเคราะห์รายบริษัทบนชุดข้อมูลเดิม (Dataset 1):**
  ```bash
  python lime33_adapter_panel.py -d 1 --issuer A
  ```

* **รันทดสอบความแม่นยำในการตรวจจับ Default ด้วยวิธี A, B, C:**
  ```bash
  python evaluate_methods_abc.py
  ```

---

## 📊 7. รายการ 30 ปัจจัยเสี่ยงในการประเมิน (30 Determinants)

แบบจำลองใช้ตัวแปรสำคัญ 30 ปัจจัยที่ครอบคลุม 5 มิติความเสี่ยง:

1. **สภาพคล่องการซื้อขาย (Liquidity & Trading)**:
   * `amihud_monthly`, `adj_illiq_kz`, `percent_zero_days`, `zero_days`, `n_days`
2. **อัตราส่วนทางการเงินและความสามารถชำระหนี้ (Financial Ratios & Solvency)**:
   * `ROA`, `ROE`, `DE`, `CurrentRatio`, `QuickRatio`, `CashRatio`, `EBITtoTA`, `REtoTA`, `WorkingCapitaltoTA`, `TDTA`, `LTDtoTA`, `STDtoTA`, `cf_Interestcoverageratio`, `acc_DebtServiceCoverageRatio`
3. **ขนาดและอายุของบริษัท (Scale & Age)**:
   * `lnTotalAssets` ($\ln(	ext{Total Assets})$), `lnAge` ($\ln(	ext{Age in Years})$)
4. **ภาวะเศรษฐกิจมหภาค (Macroeconomic Indicators)**:
   * `Policyrate` (อัตราดอกเบี้ยนโยบาย), `GDPgrowth` (การเติบโตของ GDP), `UnemploymentratemodeledILOe` (อัตราการว่างงาน)
5. **ธรรมาภิบาลและความยั่งยืน (ESG & Corporate Governance)**:
   * `ESGScore`, `GovernancePillarScore`, `EnvironmentalPillarScore`, `SocialPillarScore`, `IndependentBoardMembers`, `AverageBoardTenure`

---

## 🛡️ 8. ผลการประเมินความแม่นยำวิธี A, B, C (Accuracy Benchmark)

ระบบประเมินความเสี่ยงด้วยกลไก 3 ชั้น เพื่อปิดจุดบกพร่องของการใช้เกณฑ์ตัดความน่าจะเป็นเพียงค่าเดียว:
* **วิธี A (Level Rule)**: ตัดเกณฑ์ $	ext{PD} \ge 	ext{Threshold}$ (ความจุคิวตรวจ 5%)
* **วิธี B (Fragility Rule)**: วิเคราะห์ความเปราะบางหากตัวแปรเดี่ยวขยับเพียง $\le 1.0\,	ext{SD}$ แล้วข้ามเส้นเตือนภัย
* **วิธี C (Masked Distress Rule)**: ดักจับบริษัทที่กำไรสะสมและเงินสดอยู่ในกลุ่ม $10\%$ ต่ำสุดของตลาด (Bottom Decile)

### ตารางเปรียบเทียบผลการตรวจจับจริง:

| วิธีการคัดกรองความเสี่ยง | บริษัทที่เตือน | **ตรวจจับได้จริง** | **Recall (%)** | Precision (%) | % คิวตรวจในตลาด |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **[Dataset 1: ชุดข้อมูลเดิม (293 บริษัท, เกิดเหตุการณ์จริง 8 บริษัท)]** | | | | | |
| \quad วิธี A (Level Rule: $	ext{PD} \ge 	ext{Thr}$) | 15 บริษัท | **4 / 8** | 50.0% | 26.7% | 5.1% |
| \quad วิธี B (Fragility: Shock $\le 1.0\,	ext{SD}$) | 8 บริษัท | **2 / 8** | 25.0% | 25.0% | 2.7% |
| \quad วิธี C (Masked Distress: Bottom 10%) | 12 บริษัท | **4 / 8** | 50.0% | 33.3% | 4.1% |
| \quad **รวมทุกวิธี A + B + C** | **28 บริษัท** | **8 / 8** | **100.0%** | **28.6%** | **9.6%** |
| | | | | | |
| **[Dataset 2: ชุดข้อมูลใหม่ (941 บริษัท, เกิดวิกฤต/ปรับหนี้จริง 31 บริษัท)]** | | | | | |
| \quad วิธี A (Level Rule: $	ext{PD} \ge 	ext{Thr}$) | 48 บริษัท | **8 / 31** | 25.8% | 16.7% | 5.1% |
| \quad วิธี B (Fragility: Shock $\le 1.0\,	ext{SD}$) | 21 บริษัท | **3 / 31** | 9.7% | 14.3% | 2.2% |
| \quad วิธี C (Masked Distress: Bottom 10%) | 39 บริษัท | **5 / 31** | 16.1% | 12.8% | 4.1% |
| \quad **รวมทุกวิธี A + B + C** | **106 บริษัท** | **15 / 31** | **48.4%** | **14.2%** | **11.3%** |

*หมายเหตุ: บน Dataset 2 รายชื่อ 15 บริษัทที่ตรวจจับได้ประกอบด้วยเคสสำคัญ เช่น `STARK`, `ALL`, `ACAP`, `NWR`, `CHO`, `A`, `PRIME`, `EP`, `PF`, `JCK`, `TPOLY`, `TTCL`, `CV`, `B`, `JTS` สำหรับ 16 บริษัทที่ยังไม่ถูก Flag ในงวดล่าสุดส่วนใหญ่เป็นบริษัทที่ได้ผ่านกระบวนการฟื้นฟูกิจการหรือเพิ่มทุนเสร็จสิ้นแล้วในอดีต (เช่น `THAI`, `EA`, `ITD`)*

---

## 📖 9. เอกสารคู่มือฉบับเต็ม PDF (Documentation)

สำหรับผู้ที่ต้องการศึกษาเชิงลึก สามารถเปิดอ่านเอกสารคู่มือฉบับเต็มความยาว **50 หน้า** ได้ที่:
* 📄 **คู่มือ PDF ภาษาไทย**: [`lime33.pdf`](lime33.pdf) (ขนาด 5.88 MB)
* 📝 **ไฟล์ต้นฉบับ LaTeX**: [`lime33.tex`](lime33.tex)

**เนื้อหาภายในคู่มือประกอบด้วย**:
1. ตารางเปรียบเทียบระบบเดิม vs ระบบใหม่ DataAdapter
2. พจนานุกรมข้อมูลและโครงสร้างทั้ง 95 ฟิลด์ของตาราง `ibond_33features_panel_941firm`
3. แผนผังขั้นตอนอัลกอริทึม (Algorithm Flowchart - TikZ)
4. บัญชีรายชื่อบริษัททั้ง 941 บริษัท พร้อมช่วงเวลาเริ่มต้น-สิ้นสุด และ % ความครบถ้วนของข้อมูล
5. แกลเลอรีภาพผลการวิเคราะห์ LIME & SHAP ของ 20 บริษัทตัวอย่าง
6. ซอร์สโค้ดฉบับเต็มของ `data_adapter.py` และ `lime33_adapter_panel.py`

---
**พัฒนาโดย**: ทีมวิจัยระบบเตือนภัยล่วงหน้าตลาดตราสารหนี้ไทย (Thai Credit EWS Project)
