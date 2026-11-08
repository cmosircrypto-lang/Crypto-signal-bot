"""ثبت سیگنال‌ها و نتیجه‌ی آن‌ها در فایل اکسل جداگانه برای هر ربات."""
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import jdatetime
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill

TEHRAN = ZoneInfo("Asia/Tehran")
HEADERS = ["نام ارز", "تاریخ شمسی", "ساعت ایران", "جهت",
           "قیمت ورود", "TP", "SL", "نتیجه"]
WIDTHS = [22, 14, 14, 10, 14, 14, 14, 30]
RESULT_COL = 8


def excel_filename(bot_name):
    # crypto-signal-bot -> signals_bot.xlsx ، crypto-signal-bot-v10 -> signals_bot-v10.xlsx
    return "signals_" + bot_name.replace("crypto-signal-bot", "bot") + ".xlsx"


def _jalali(dt):
    return jdatetime.datetime.fromgregorian(datetime=dt).strftime("%Y/%m/%d")


def _open(path):
    if os.path.exists(path):
        wb = load_workbook(path)
        return wb, wb.active
    wb = Workbook()
    ws = wb.active
    ws.title = "Signals"
    ws.sheet_view.rightToLeft = True
    ws.append(HEADERS)
    for i, w in enumerate(WIDTHS, 1):
        col = ws.cell(row=1, column=i)
        col.font = Font(bold=True, color="FFFFFF")
        col.fill = PatternFill("solid", fgColor="305496")
        col.alignment = Alignment(horizontal="center")
        ws.column_dimensions[col.column_letter].width = w
    ws.freeze_panes = "A2"
    return wb, ws


def log_signal(bot_name, symbol, direction, entry, tp, sl):
    """ثبت یک سیگنال جدید (ستون نتیجه خالی می‌ماند تا معامله بسته شود)."""
    try:
        now = datetime.now(TEHRAN)
        path = excel_filename(bot_name)
        wb, ws = _open(path)
        ws.append([
            symbol,
            _jalali(now),
            now.strftime("%H:%M:%S"),
            "لانگ" if direction == "LONG" else "شرت",
            float(entry), float(tp), float(sl),
            "",
        ])
        wb.save(path)
    except Exception as e:
        print(f"⚠️ Excel log_signal error ({bot_name}): {e}")


def log_result(bot_name, symbol, result, closed_ms):
    """نتیجه (TP یا SL) با تاریخ و ساعت شمسی ایران را در آخرین ردیف باز همان نماد می‌نویسد."""
    try:
        path = excel_filename(bot_name)
        if not os.path.exists(path):
            return
        wb = load_workbook(path)
        ws = wb.active
        for row in range(ws.max_row, 1, -1):
            if ws.cell(row=row, column=1).value == symbol and not ws.cell(row=row, column=RESULT_COL).value:
                dt = datetime.fromtimestamp(closed_ms / 1000, TEHRAN)
                cell = ws.cell(row=row, column=RESULT_COL)
                cell.value = f"{result} | {_jalali(dt)} {dt.strftime('%H:%M:%S')}"
                cell.fill = PatternFill("solid", fgColor="C6EFCE" if result == "TP" else "FFC7CE")
                wb.save(path)
                return
    except Exception as e:
        print(f"⚠️ Excel log_result error ({bot_name}): {e}")
