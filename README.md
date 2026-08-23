# ระบบเตือนภัยล่วงหน้าและอธิบายความเสี่ยงเครดิตตราสารหนี้ไทยด้วย AI (Project LIME)
### Explainable AI (LIME & SHAP) Early Warning System for Thai Corporate Bond Defaults

> ระบบปัญญาประดิษฐ์อธิบายผลความเสี่ยงเครดิต (XAI) สำหรับการประเมินโอกาสผิดนัดชำระหนี้และการปรับโครงสร้างหนี้ของบริษัทในตลาดทุนไทย ครอบคลุม 30 ปัจจัยเสี่ยงทางการเงิน สภาพคล่อง มหภาค และ ESG พร้อมสถาปัตยกรรมเชื่อมต่อข้อมูล 2 ชุด (DataAdapter), ระบบสแกน **Rolling Window 12 เดือน**, และโมเดลจำแนกกลุ่ม **Segmented Models (Bond vs mai)**

---

## 📑 สารบัญ (Table of Contents)
1. [ภาพรวมของโครงการ (Overview)](#-1-ภาพรวมของโครงการ-overview)
2. [สถาปัตยกรรมระบบและจุดเด่นหลัก (Key Features)](#-2-สถาปัตยกรรมระบบและจุดเด่นหลัก-key-features)
3. [ผลการทดสอบเปรียบเทียบกลยุทธ์เตือนภัยบน Dataset 2 (Benchmark Results)](#-3-ผลการทดสอบเปรียบเทียบกลยุทธ์เตือนภัยบน-dataset-2-benchmark-results)
4. [โครงสร้างไฟล์และโฟลเดอร์ในโครงการ (Project Structure)](#-4-โครงสร้างไฟล์และโฟลเดอร์ในโครงการ-project-structure)
5. [โครงสร้างฐานข้อมูล SQLite (`lime_credit.db`)](#-5-โครงสร้างฐานข้อมูล-sqlite-lime_creditdb)
6. [การติดตั้งและข้อกำหนดของระบบ (Installation & Setup)](#-6-การติดตั้งและข้อกำหนดของระบบ-installation--setup)
7. [คู่มือการใช้งานคำสั่ง (Command-Line Reference)](#-7-คู่มือการใช้งานคำสั่ง-command-line-reference)
8. [รายการ 30 ปัจจัยเสี่ยงในการประเมิน (30 Determinants)](#-8-รายการ-30-ปัจจัยเสี่ยงในการประเมิน-30-determinants)
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
2. **โหมดสแกนเตือนภัยต่อเนื่อง (Rolling Window 12 / 24 Months)**:
   - แจ้งเตือนทั้งบริษัทที่มีความเสี่ยงสูงในงวดปัจจุบัน และบริษัทที่เคยมีสัญญาณเตือนติดคิวสะสมในรอบ 12–24 เดือนที่ผ่านมา เพื่อดักจับบริษัทที่อยู่ในกระบวนการปรับโครงสร้างหนี้ (เช่น `EA`, `ITD`, `RICHY`, `SQ`, `ECF`, `MJD`, `NRF`)
3. **การแบ่งกลุ่มวิเคราะห์เฉพาะทาง (Segmented Universe Models)**:
   - แยกตัดเกณฑ์ Review Threshold ระหว่าง **กลุ่มผู้ออกตราสารหนี้ (Bond Issuers)** และ **กลุ่มหุ้นขนาดเล็ก (Small-Cap mai)** ป้องกันไม่ให้สภาพคล่องหุ้นเล็กเบียดบังคิวตรวจของหุ้นกู้
4. **การอธิบายผลแบบประกบคู่ (Repeated LIME + Exact Tree SHAP)**:
   - **Repeated-Seed LIME**: สุ่มรบกวน 5,000 จุด ทำซ้ำ 8 Seed พร้อมคำนวณแถบความเชื่อมั่น
   - **Exact Additive Tree SHAP**: คำนวณค่าน้ำหนักเสริมความเสี่ยงจริงตามทฤษฎีเกม
   - **Distance from Median**: แสดงระยะห่างของตัวแปรเทียบกับค่ามัธยฐานของตลาดในหน่วย SD

---

## 🏆 3. ผลการทดสอบเปรียบเทียบกลยุทธ์เตือนภัยบน Dataset 2 (Benchmark Results)

ตารางเปรียบเทียบผลการทดสอบบนชุดข้อมูล 941 บริษัท (มีบริษัทที่เกิดวิกฤต/ปรับโครงสร้างหนี้ 31 บริษัท แบ่งเป็นกลุ่มผู้ออกหุ้นกู้ 24 บริษัท และหุ้นขนาดเล็ก 7 บริษัท):

| กลยุทธ์การตรวจจับความเสี่ยง (Strategy) | บริษัทที่เตือน (Flagged) | **ตรวจจับได้รวม** | **Recall รวม (%)** | Precision (%) | **ตรวจจับกลุ่มหุ้นกู้** | **Recall หุ้นกู้ (%)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Snapshot งวดเดียว (Baseline Pooled)** | 87 บริษัท | **13 / 31** | **41.9%** | 14.9% | 10 / 24 | 41.7% |
| **2. Snapshot แยกกลุ่ม (Segmented Models)** | 86 บริษัท | **14 / 31** | **45.2%** | 16.3% | 11 / 24 | 45.8% |
| **3. Rolling Window 12 เดือน (Pooled)** | 223 บริษัท | **23 / 31** | **74.2%** | 10.3% | 18 / 24 | 75.0% |
| **4. Segmented + Rolling 12 เดือน (แนะนำ)** | **243 บริษัท** | **24 / 31** | **77.4%** | **9.9%** | **19 / 24** | **79.2%** |
| **5. Segmented + Rolling 24 เดือน (เต็มรอบหนี้)** | **300 บริษัท** | **26 / 31** | **83.9%** | **8.7%** | **21 / 24** | **87.5%** |
| **6. Dynamic Lead-Time (12 ด. ก่อนเกิดเหตุจริง)** | 241 บริษัท | **22 / 31** | **71.0%** | 9.1% | 20 / 24 | 83.3% |

### 💡 บทวิเคราะห์ผลลัพธ์:
* **การใช้ Rolling Window 12 เดือน (กลยุทธ์ที่ 4)** ช่วยเพิ่มค่า Recall จาก $41.9\%$ พุ่งขึ้นเป็น **$77.4\%$** (ดักจับได้ 24 จาก 31 บริษัท) โดยจับเคสสำคัญได้ครบถ้วน เช่น `A`, `ALL`, `CHO`, `CV`, `ECF`, `EP`, `GRAND`, `ITD`, `JCK`, `JTS`, `MJD`, `NRF`, `NWR`, `PF`, `POWER`, `PRIME`, `SQ`, `STARK`, `TPOLY`, `TTCL`
* **การขยายเป็น Rolling 24 เดือน (กลยุทธ์ที่ 5)** สามารถจับบริษัทขนาดใหญ่ที่ปรับโครงสร้างหนี้ระยะยาวได้เพิ่มขึ้น เช่น `EA`, `RICHY` ทำให้ Recall พุ่งแตะ **$83.9\%$** (และกลุ่มหุ้นกู้สูงถึง **$87.5\%$**)

---

## 📂 4. โครงสร้างไฟล์และโฟลเดอร์ในโครงการ (Project Structure)

```
iBond_LIME/
├── lime33_adapter_panel.py     # โปรแกรมหลัก (รองรับ --rolling-window และ --segmented)
├── data_adapter.py             # โมดูล DataAdapter เชื่อมต่อ SQLite พร้อมระบบ Auto-Unzip
├── evaluate_methods_abc.py     # สคริปต์ประเมินความแม่นยำวิธี A, B, C เปรียบเทียบทั้ง 2 ชุดข้อมูล
├── batch_run_20_issuers.py     # สคริปต์ประมวลผลและสร้างรูปภาพของ 20 บริษัทตัวอย่าง
├── a_approach.py               # โมดูลประเมินความเสี่ยง 3 ชั้น
├── firm_shock_panel.py         # โมดูลวิเคราะห์ความเปราะบางของปัจจัย
├── cmdf_tree_classify.py       # โมดูลสร้างโมเดล Machine Learning
├── lime_credit.db.zip          # ไฟล์ฐานข้อมูล SQLite บีบอัด (23.8 MB คลายซิปอัตโนมัติ)
├── lime33.tex                  # ซอร์สโค้ดคู่มือ LaTeX ภาษาไทยฉบับสมบูรณ์ (50 หน้า)
├── lime33.pdf                  # คู่มือการใช้งานฉบับสมบูรณ์ PDF (50 หน้า)
├── requirements.txt            # รายการไลบรารี Python ที่ต้องติดตั้ง
├── README.md                   # เอกสารคู่มือโครงการภาษาไทย
└── tex_out/
    ├── lime_jpg/               # โฟลเดอร์เก็บภาพผลการอธิบายความเสี่ยง 3 แผง (.jpg)
    ├── lime_figs/              # โฟลเดอร์เก็บภาพผลการวิเคราะห์ (.png)
    ├── summary_941_firms.csv   # สรุปรายชื่อ 941 บริษัท, ช่วงเวลา, และ % ความครบถ้วนของฟีเจอร์
    └── comparison_rolling_segmented.csv # ตารางเปรียบเทียบผลลัพธ์ทั้ง 6 กลยุทธ์
```

---

## 🗄️ 5. โครงสร้างฐานข้อมูล SQLite (`lime_credit.db`)

| ชื่อตาราง (Table Name) | จำนวนแถว | จำนวนคอลัมน์ | รายละเอียดข้อมูล |
| :--- | :---: | :---: | :--- |
| **`ibond_33features_panel`** | 16,986 แถว | 61 คอลัมน์ | ชุดข้อมูลเดิม: ผู้ออกตราสารหนี้ 219–293 บริษัท |
| **`ibond_33features_panel_941firm`** | 187,007 แถว | 95 คอลัมน์ | ชุดข้อมูลใหม่: 941 บริษัททั้ง SET/mai ครอบคลุมถึง ส.ค. 2026 |
| **`ibond_default_payment`** | 50 แถว | 12 คอลัมน์ | ทะเบียนประวัติการผิดนัดชำระหนี้ตราสารหนี้ทางการของ ThaiBMA |
| **`firm_issuer_mapping`** | 985 แถว | 6 คอลัมน์ | ตารางจับคู่ Ticker Symbol กับรหัสผู้ออกตราสารหนี้ |
| **`ibond_issuer`** | 678 แถว | 8 คอลัมน์ | ข้อมูลบริษัทและหมวดธุรกิจของผู้ออกตราสารหนี้ |

---

## 💻 6. การติดตั้งและข้อกำหนดของระบบ (Installation & Setup)

```bash
git clone https://github.com/kkabinsky/iBond_LIME.git
cd iBond_LIME
pip install -r requirements.txt
```

---

## 🚀 7. คู่มือการใช้งานคำสั่ง (Command-Line Reference)

### 7.1 รันแบบเมนูเลือกหน้าจอ (Interactive Menu Mode)
```bash
python lime33_adapter_panel.py
```

### 7.2 รันแบบ Segmented + Rolling Window (แนะนำ):
* **วิเคราะห์รายบริษัทด้วยโมเดล Segmented และ Rolling 12 เดือน:**
  ```bash
  python lime33_adapter_panel.py -d 2 --segmented -w 12 --issuer PTT
  python lime33_adapter_panel.py -d 2 --segmented -w 12 --issuer ITD
  python lime33_adapter_panel.py -d 2 --segmented -w 12 --issuer EA
  ```

* **วิเคราะห์ทุกบริษัทในกลุ่มเสี่ยงสูง (High-Risk Queue):**
  ```bash
  python lime33_adapter_panel.py -d 2 --segmented -w 12 --all-high-risk
  ```

* **รันประเมินความแม่นยำเปรียบเทียบทุกกลยุทธ์:**
  ```bash
  python benchmark_rolling_segmented.py
  ```

---

## 📊 8. รายการ 30 ปัจจัยเสี่ยงในการประเมิน (30 Determinants)

1. **สภาพคล่องการซื้อขาย**: `amihud_monthly`, `adj_illiq_kz`, `percent_zero_days`, `zero_days`, `n_days`
2. **อัตราส่วนทางการเงินและความสามารถชำระหนี้**: `ROA`, `ROE`, `DE`, `CurrentRatio`, `QuickRatio`, `CashRatio`, `EBITtoTA`, `REtoTA`, `WorkingCapitaltoTA`, `TDTA`, `LTDtoTA`, `STDtoTA`, `cf_Interestcoverageratio`, `acc_DebtServiceCoverageRatio`
3. **ขนาดและอายุของบริษัท**: `lnTotalAssets`, `lnAge`
4. **ภาวะเศรษฐกิจมหภาค**: `Policyrate`, `GDPgrowth`, `UnemploymentratemodeledILOe`
5. **ธรรมาภิบาลและความยั่งยืน (ESG)**: `ESGScore`, `GovernancePillarScore`, `EnvironmentalPillarScore`, `SocialPillarScore`, `IndependentBoardMembers`, `AverageBoardTenure`

---

## 📖 9. เอกสารคู่มือฉบับเต็ม PDF (Documentation)
* 📄 **คู่มือ PDF ภาษาไทย**: [`lime33.pdf`](lime33.pdf) (50 หน้า)
* 📝 **ไฟล์ต้นฉบับ LaTeX**: [`lime33.tex`](lime33.tex)
