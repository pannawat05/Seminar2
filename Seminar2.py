import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ตั้งค่าฟอนต์
plt.rcParams['font.family'] = 'tahoma'

file_path = "ข้อมูลย้อนหลังของ ดัชนี SET.csv"
next_day = 32

try:
    # โหลดข้อมูล (ใส่ encoding ให้รองรับภาษาไทย)
    # และใช้ thousands=',' เพื่อจัดการเครื่องหมาย , ในตัวเลข
    df = pd.read_csv(file_path, encoding="utf-8-sig", thousands=',')

    # ตรวจสอบว่าคอลัมน์ที่ต้องการมีอยู่หรือไม่
    required_cols = ['วันที่', 'ล่าสุด']
    if not all(col in df.columns for col in required_cols):
        print("❌ ข้อผิดพลาด: ไม่พบคอลัมน์ 'วันที่' หรือ 'ล่าสุด' ในไฟล์ CSV")
        print(f"คอลัมน์ที่มีอยู่ในไฟล์: {df.columns.tolist()}")
    else:
        # เตรียมข้อมูลสำหรับ Regression
        # 'ล่าสุด' ถูกแปลงเป็นตัวเลขแล้วโดย thousands=','
        # 'วันที่' ก็เป็นตัวเลขอยู่แล้ว
        s = df['ล่าสุด'].to_numpy()
        d = df['วันที่'].to_numpy()
        
        # ตรวจสอบว่ามีข้อมูลเพียงพอหรือไม่
        if len(d) < 2 or len(s) < 2:
            print("❌ ข้อมูลไม่เพียงพอสำหรับทำ regression (ต้องมีอย่างน้อย 2 แถวที่ถูกต้อง)")
        else:
            # สร้างสมการพหุนาม (polynomial regression)
            poly = np.poly1d(np.polyfit(d, s, 1))

            # แสดงผล
            print(f"✅ สมการแนวโน้ม:\n{poly}")
            predict = poly(next_day)
            print(f"➡️ พยากรณ์หุ้นวันที่ {next_day} คือ {predict.round(2)} ")

            # วาดกราฟ
            plt.plot(d, s, 'o', label="ราคาหุ้นจริง")
            plt.plot(d, poly(d), label="เส้นแนวโน้ม")
            plt.scatter(next_day, predict, color="red", label="ค่าที่พยากรณ์")
            plt.xlabel("วันที่")
            plt.ylabel("ราคาหุ้น")
            plt.legend()
            plt.grid(True)
            plt.show()

except FileNotFoundError:
    print(f"❌ ไม่พบไฟล์ '{file_path}'")
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดที่ไม่คาดคิด: {str(e)}")