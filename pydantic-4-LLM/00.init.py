import os
import sqlite3

# 程序用途：
# 1. 在customer_emails目录下生成三个邮件文本
# 2. 初始化sqlite数据库，生成一张空的issue表

# 创建一个用于存放邮件文件的目录
EMAIL_DIR = "./customer_emails"
DATABASE_NAME = "customer_issues.db"

os.makedirs(EMAIL_DIR, exist_ok=True)

# 模拟邮件内容
email_contents = {
    "email_1.txt": """
发件人：李女士 <li.nvshi@example.com>
主题：XYZ 路由器 Wi-Fi 问题

尊敬的客服团队，

我的 XYZ 路由器突然无法连接 Wi-Fi 了。指示灯显示正常，但我的所有设备都无法找到 Wi-Fi 信号。我已经尝试重启路由器和我的手机，但都没有解决问题。这给我带来了很大的不便，希望能够尽快解决，优先级比较高。谢谢。

李女士
""",
    "email_2.txt": """
发件人：王先生 <wang.xs@example.com>
主题：关于我的新 ABC 智能电视的色彩问题

您好，

我上周购买的 ABC 智能电视，在使用过程中发现屏幕颜色有点偏绿，看起来很不舒服。这个是新产品，希望能给个解决方案。感觉是个中等优先级的问题吧。

谢谢，
王先生
""",
    "email_3.txt": """
发件人：张小姐 <zhang.xj@example.com>
主题：紧急！我的DEF手机完全无法开机

客服您好，

我的DEF手机今天早上突然就无法开机了，充了电也没反应。里面有很多重要资料，我急需使用。请立即帮我处理！这个是紧急事件！

张小姐
"""
}

def create_database_table():
    """
    创建 SQLite 数据库和表。
    """
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS issues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_name TEXT NOT NULL,
            product TEXT NOT NULL,
            issue_description TEXT NOT NULL,
            priority TEXT NOT NULL,
            assigned_department TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print(f"数据库 '{DATABASE_NAME}' 和 'issues' 表已准备就绪。")


# 将内容写入文件
for filename, content in email_contents.items():
    with open(os.path.join(EMAIL_DIR, filename), "w", encoding="utf-8") as f:
        f.write(content)

# 创建客户问题数据库
create_database_table()