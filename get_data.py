import os
import json
import time
import base64
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from xhtml2pdf import pisa
# 替换 cloudscraper 为 curl_cffi
from curl_cffi import requests as cffi_requests


#从https://www.foodnetwork.com/recipes爬取菜谱构建pdf格式，模拟文件输入

# ================= 配置区 =================
OUTPUT_DIR = "food_network_pdfs"
MAX_RECIPES = 3  
REQUEST_DELAY = 3  

# 【关键】代理配置！如果你在国内，必须配置代理，否则必 403。
# 请将 'http://127.0.0.1:7890' 替换为你本地真实的代理地址和端口（如 Clash/V2ray 的 HTTP 端口）
# 如果你确定不需要代理（比如你在海外服务器），请将 PROXIES 设为 None
PROXIES = {
    "http": "http://127.0.0.1:8890",
    "https": "http://127.0.0.1:8890"
}

# PROXIES = None  # 如果在海外或不需要代理，使用这行

# 测试用的具体菜谱链接
TEST_URLS = [
    "https://www.foodnetwork.com/recipes/ree-drummond/lighter-chicken-parmesan-3588766",
    "https://www.foodnetwork.com/recipes/chicken-parmesan-8536888",
    "https://www.foodnetwork.com/recipes/tyler-florence/chicken-parmesan-recipe-1951852",
]

def get_session():
    """
    创建一个模拟真实 Chrome 浏览器的会话
    impersonate="chrome110" 会完美模拟 Chrome 110 版本的底层网络指纹
    """
    return cffi_requests.Session(impersonate="chrome110")

def download_image_as_base64(image_url, session):
    """下载图片并转换为 base64 编码"""
    try:
        # 使用 curl_cffi 下载图片
        response = session.get(image_url, proxies=PROXIES, timeout=15)
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')
            img_format = 'jpeg'
            if 'png' in content_type: img_format = 'png'
            elif 'gif' in content_type: img_format = 'gif'
            elif 'webp' in content_type: img_format = 'webp'
            
            img_base64 = base64.b64encode(response.content).decode('utf-8')
            return f"data:image/{img_format};base64,{img_base64}"
    except Exception as e:
        print(f"    [!] 图片下载失败: {e}")
    return None

def extract_recipe_data(url, session):
    """核心抽取逻辑"""
    print(f"[*] 正在爬取: {url}")
    try:
        # 使用 curl_cffi 发起请求，带上代理
        response = session.get(url, proxies=PROXIES, timeout=15)
        response.raise_for_status()
    except Exception as e:
        print(f"[!] 请求失败: {e}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    recipe_data = None

    # 1. 尝试从 JSON-LD 中提取结构化数据
    scripts = soup.find_all('script', type='application/ld+json')
    for script in scripts:
        try:
            data = json.loads(script.string)
            if isinstance(data, dict) and '@graph' in data:
                data = data['@graph']
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get('@type') == 'Recipe':
                        recipe_data = item
                        break
            elif isinstance(data, dict) and data.get('@type') == 'Recipe':
                recipe_data = data
            
            if recipe_data:
                break
        except (json.JSONDecodeError, TypeError):
            continue

    if not recipe_data:
        print("[!] 未找到 JSON-LD 结构化数据，跳过该页面。")
        return None

    # 2. 数据清洗
    title = recipe_data.get('name', 'Unknown Recipe')
    ingredients = recipe_data.get('recipeIngredient', [])
    
    raw_instructions = recipe_data.get('recipeInstructions', [])
    instructions = []
    if isinstance(raw_instructions, str):
        instructions = [step.strip() for step in raw_instructions.split('\n') if step.strip()]
    elif isinstance(raw_instructions, list):
        for step in raw_instructions:
            if isinstance(step, dict) and 'text' in step:
                instructions.append(step['text'])
            elif isinstance(step, str):
                instructions.append(step)

    # 3. 提取并下载图片
    image_url = None
    image_data = recipe_data.get('image')
    if image_data:
        if isinstance(image_data, str):
            image_url = image_data
        elif isinstance(image_data, list) and len(image_data) > 0:
            image_url = image_data[0] if isinstance(image_data[0], str) else image_data[0].get('url')
        elif isinstance(image_data, dict):
            image_url = image_data.get('url')
    
    image_base64 = None
    if image_url:
        print(f"    下载图片...")
        image_base64 = download_image_as_base64(image_url, session)
        if image_base64:
            print(f"    图片下载成功!")

    return {
        'title': title,
        'ingredients': ingredients,
        'instructions': instructions,
        'url': url,
        'image': image_base64
    }

def generate_html(recipe):
    """渲染 HTML"""
    ingredients_html = "\n".join([f"<li>{ing}</li>" for ing in recipe['ingredients']])
    instructions_html = "\n".join([f"<li>{step}</li>" for step in recipe['instructions']])
    
    image_html = ""
    if recipe['image']:
        image_html = f'''
        <div style="text-align: center; margin: 20px 0;">
            <img src="{recipe['image']}" style="max-width: 100%; height: auto; max-height: 400px; border: 1px solid #ddd; border-radius: 8px;" />
        </div>
        '''
    else:
        image_html = '<div style="text-align: center; margin: 20px 0; padding: 50px; background: #f5f5f5;"><p>图片暂不可用</p></div>'
    
    html_template = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            @page {{ size: A4; margin: 2cm; }}
            body {{ font-family: Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; }}
            h1 {{ color: #d32323; font-size: 24px; border-bottom: 2px solid #eee; padding-bottom: 10px; text-align: center; }}
            h2 {{ color: #555; font-size: 18px; margin-top: 20px; border-left: 4px solid #d32323; padding-left: 10px; }}
            ul {{ background: #f9f9f9; padding: 15px 15px 15px 30px; margin: 0; border-left: 3px solid #d32323; }}
            li {{ margin-bottom: 8px; }}
            ol li {{ margin-bottom: 15px; }}
            .footer {{ margin-top: 30px; font-size: 12px; color: #999; text-align: center; border-top: 1px solid #eee; padding-top: 10px; }}
        </style>
    </head>
    <body>
        <h1>{recipe['title']}</h1>
        {image_html}
        <h2>配料 (Ingredients)</h2>
        <ul>{ingredients_html}</ul>
        <h2>制作步骤 (Instructions)</h2>
        <ol>{instructions_html}</ol>
        <div class="footer">Source: {recipe['url']}</div>
    </body>
    </html>
    """
    return html_template

def save_as_pdf(html_content, output_path):
    """生成 PDF"""
    try:
        with open(output_path, "w+b") as pdf_file:
            pisa_status = pisa.CreatePDF(html_content, dest=pdf_file)
        return pisa_status.err
    except Exception as e:
        print(f"[!] PDF 生成错误: {e}")
        return True

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session = get_session()
    
    target_urls = TEST_URLS[:MAX_RECIPES]
    print(f"[*] 准备爬取 {len(target_urls)} 个菜谱...\n")
    
    success_count = 0
    for url in target_urls:
        recipe = extract_recipe_data(url, session)
        if recipe:
            html_content = generate_html(recipe)
            safe_filename = "".join([c for c in recipe['title'] if c.isalnum() or c in (' ', '-', '_')]).rstrip()
            pdf_path = os.path.join(OUTPUT_DIR, f"{safe_filename}.pdf")
            
            if not save_as_pdf(html_content, pdf_path):
                print(f"[+] 成功导出 PDF: {pdf_path}")
                success_count += 1
            else:
                print(f"[!] PDF 生成失败: {pdf_path}")
        
        time.sleep(REQUEST_DELAY)

    print(f"\n[*] 任务完成！成功生成 {success_count}/{len(target_urls)} 个 PDF 文件")

if __name__ == "__main__":
    main()