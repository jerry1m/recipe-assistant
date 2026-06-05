import os
import json
import time
import base64
import re
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from xhtml2pdf import pisa
from curl_cffi import requests as cffi_requests

# ================= 配置区 =================
OUTPUT_DIR = "foodology_pdfs"
MAX_RECIPES = 5
REQUEST_DELAY = 2
DEBUG_MODE = True  # 开启调试，查看提取到的原始数据

PROXIES = None
# PROXIES = {"http": "http://127.0.0.1:8890", "https": "http://127.0.0.1:8890"}

TEST_URLS = [
    "https://www.foodologygeek.com/focaccia-bread-recipe/",
    "https://www.foodologygeek.com/lomi-lomi-salmon/",
    "https://www.foodologygeek.com/glazed-spiral-slice-ham-recipe/",
    "https://www.foodologygeek.com/homemade-peanut-butter-dog-treats/",
   " https://www.foodologygeek.com/tomato-garlic-rosemary-focaccia/"
]

def get_session():
    return cffi_requests.Session(impersonate="chrome110")

def debug_print(title, data):
    if DEBUG_MODE:
        print(f"\n{'='*60}\n📋 {title}: {data}\n{'='*60}")

def parse_time(time_str):
    if not time_str: return ""
    if isinstance(time_str, str) and time_str.startswith('PT'):
        hours = re.search(r'(\d+)H', time_str)
        minutes = re.search(r'(\d+)M', time_str)
        days = re.search(r'(\d+)D', time_str)
        result = []
        if days: result.append(f"{days.group(1)} d")
        if hours: result.append(f"{hours.group(1)} hrs")
        if minutes: result.append(f"{minutes.group(1)} mins")
        return " ".join(result) if result else time_str
    return time_str

def download_image_as_base64(image_url, session):
    if not image_url: return None
    try:
        if image_url.startswith('/'):
            image_url = "https://www.foodologygeek.com" + image_url
        response = session.get(image_url, proxies=PROXIES, timeout=15)
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')
            img_format = 'jpeg'
            if 'png' in content_type: img_format = 'png'
            elif 'webp' in content_type: img_format = 'webp'
            img_base64 = base64.b64encode(response.content).decode('utf-8')
            return f"data:image/{img_format};base64,{img_base64}"
    except Exception as e:
        print(f"    [!] 图片下载失败: {e}")
    return None

def extract_nutrition_from_bottom(soup):
    """提取底部 WP Recipe Maker 的营养信息"""
    nutrition_data = {
        'calories': '', 'protein': '', 'fat': '', 'saturated_fat': '',
        'carbohydrates': '', 'fiber': '', 'sugar': '', 'sodium': '', 'cholesterol': ''
    }
    # WP Recipe Maker 的营养信息通常在 wprm-recipe-nutrition 容器中
    nutrition_container = soup.find(class_=lambda x: x and 'wprm-recipe-nutrition' in x)
    if nutrition_container:
        text = nutrition_container.get_text()
        patterns = {
            'calories': r'[Cc]alories[:\s]*(\d+)',
            'protein': r'[Pp]rotein[:\s]*(\d+(?:\.\d+)?)\s*g',
            'fat': r'[Ff]at[:\s]*(\d+(?:\.\d+)?)\s*g',
            'saturated_fat': r'[Ssaturated\s*[Ff]at|Saturated Fat][: \s]*(\d+(?:\.\d+)?)\s*g',
            'carbohydrates': r'[Cc]arbohydrates?[:\s]*(\d+(?:\.\d+)?)\s*g',
            'fiber': r'[Ff]iber[:\s]*(\d+(?:\.\d+)?)\s*g',
            'sugar': r'[Ssugar|Sugars][: \s]*(\d+(?:\.\d+)?)\s*g',
            'sodium': r'[Ssodium[:\s]*(\d+(?:\.\d+)?)\s*mg',
            'cholesterol': r'[Cc]holesterol[:\s]*(\d+(?:\.\d+)?)\s*mg',
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                nutrition_data[key] = match.group(1)
    return nutrition_data

def extract_recipe_data(url, session):
    print(f"\n[*] 正在爬取: {url}")
    try:
        response = session.get(url, proxies=PROXIES, timeout=15)
        response.raise_for_status()
    except Exception as e:
        print(f"[!] 请求失败: {e}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    recipe_data = {}
    json_ld_found = False
    
    # 1. 提取 JSON-LD
    scripts = soup.find_all('script', type='application/ld+json')
    for script in scripts:
        try:
            data = json.loads(script.string)
            if isinstance(data, dict) and '@graph' in data: data = data['@graph']
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get('@type') == 'Recipe':
                        recipe_data = item; json_ld_found = True; break
            elif isinstance(data, dict) and data.get('@type') == 'Recipe':
                recipe_data = data; json_ld_found = True
            if json_ld_found: break
        except: continue

    extracted = {
        'title': 'Unknown Recipe', 'author': '', 'description': '',
        'prep_time': '', 'cook_time': '', 'total_time': '',
        'servings': '', 'category': '', 'cuisine': '',
        'ingredients': [], 'instructions': [], 'image_url': None
    }

    if json_ld_found:
        extracted['title'] = recipe_data.get('name', 'Unknown Recipe')
        extracted['description'] = recipe_data.get('description', '')
        extracted['prep_time'] = parse_time(recipe_data.get('prepTime', ''))
        extracted['cook_time'] = parse_time(recipe_data.get('cookTime', ''))
        extracted['total_time'] = parse_time(recipe_data.get('totalTime', ''))
        
        # 作者
        author = recipe_data.get('author', {})
        if isinstance(author, dict): extracted['author'] = author.get('name', '')
        elif isinstance(author, list) and author: extracted['author'] = author[0].get('name', '') if isinstance(author[0], dict) else ''
        
        # 菜系 (处理数组)
        cuisine_data = recipe_data.get('recipeCuisine', '')
        if isinstance(cuisine_data, list): extracted['cuisine'] = ', '.join(cuisine_data)
        else: extracted['cuisine'] = cuisine_data if isinstance(cuisine_data, str) else ''
        
        # 类别
        cat_data = recipe_data.get('recipeCategory', '')
        if isinstance(cat_data, list): extracted['category'] = ', '.join(cat_data)
        else: extracted['category'] = cat_data if isinstance(cat_data, str) else ''

        # 配料
        extracted['ingredients'] = recipe_data.get('recipeIngredient', [])

        # 【核心修复】处理 WP Recipe Maker 的嵌套步骤 (HowToSection)
        raw_instructions = recipe_data.get('recipeInstructions', [])
        step_counter = 1
        if isinstance(raw_instructions, str):
            extracted['instructions'] = [s.strip() for s in re.split(r'\n+', raw_instructions) if s.strip()]
        elif isinstance(raw_instructions, list):
            for item in raw_instructions:
                if isinstance(item, dict):
                    # 如果是分块 (HowToSection)
                    if item.get('@type') == 'HowToSection':
                        section_name = item.get('name', '')
                        if section_name:
                            extracted['instructions'].append(f"--- {section_name} ---")
                        for sub_step in item.get('itemListElement', []):
                            if isinstance(sub_step, dict) and 'text' in sub_step:
                                text = sub_step.get('text', '').strip()
                                if text:
                                    extracted['instructions'].append(f"Step {step_counter}: {text}")
                                    step_counter += 1
                    # 如果是普通步骤 (HowToStep)
                    elif 'text' in item:
                        text = item.get('text', '').strip()
                        if text:
                            extracted['instructions'].append(f"Step {step_counter}: {text}")
                            step_counter += 1
                elif isinstance(item, str) and item.strip():
                    extracted['instructions'].append(f"Step {step_counter}: {item.strip()}")
                    step_counter += 1

        # 图片
        image_data = recipe_data.get('image')
        if isinstance(image_data, str): extracted['image_url'] = image_data
        elif isinstance(image_data, list) and image_data:
            extracted['image_url'] = image_data[0] if isinstance(image_data[0], str) else image_data[0].get('url')
        elif isinstance(image_data, dict): extracted['image_url'] = image_data.get('url')

    # 2. 【核心修复】使用 BeautifulSoup 精准提取 WP Recipe Maker 的 DOM 元素 (覆盖 JSON-LD 的不准确值)
    
    # 提取 Servings (份量)
    servings_container = soup.find(class_="wprm-recipe-servings-container")
    if servings_container:
        servings_elem = servings_container.find(class_="wprm-recipe-servings")
        extracted['servings'] = servings_elem.get_text(strip=True) if servings_elem else servings_container.get_text(strip=True)
    else:
        # 后备：从 JSON-LD 获取
        yield_data = recipe_data.get('recipeYield', '')
        if isinstance(yield_data, list): extracted['servings'] = yield_data[0] if yield_data else ''
        else: extracted['servings'] = yield_data

    # 3. 提取底部营养信息
    nutrition = extract_nutrition_from_bottom(soup)
    # 合并 JSON-LD 中的营养信息作为后备
    json_nutrition = recipe_data.get('nutrition', {})
    if isinstance(json_nutrition, dict):
        for key in ['calories', 'protein', 'fat', 'saturated_fat', 'carbohydrates', 'fiber', 'sugar', 'sodium', 'cholesterol']:
            if not nutrition.get(key):
                val = json_nutrition.get(f"{key}Content" if key != 'calories' else 'calories', '')
                if val: nutrition[key] = str(val).replace(' g', '').replace(' mg', '').replace(' kcal', '').strip()

    # 4. 下载图片
    image_base64 = None
    if extracted.get('image_url'):
        print(f"    下载图片...")
        image_base64 = download_image_as_base64(extracted['image_url'], session)
        if image_base64: print(f"    图片下载成功!")

    # 调试输出
    debug_print("Servings (份量)", extracted['servings'])
    debug_print("Cuisine (菜系)", extracted['cuisine'])
    debug_print("步骤数量", len(extracted['instructions']))
    
    return {
        **extracted,
        'url': url,
        'image': image_base64,
        'nutrition': nutrition
    }

def generate_html(recipe):
    ingredients_html = "\n".join([f"<li>{ing}</li>" for ing in recipe['ingredients']])
    # 步骤 HTML：如果是分块标题，特殊样式
    instructions_html = ""
    for step in recipe['instructions']:
        if step.startswith("---") and step.endswith("---"):
            instructions_html += f'<li style="font-weight: bold; color: #2c5f2d; margin-top: 15px; list-style-type: none; border-bottom: 1px dashed #ccc; padding-bottom: 5px;">{step.strip("- ")}</li>'
        else:
            instructions_html += f"<li>{step}</li>"

    author_html = f'<p style="margin: 5px 0; color: #666;">👨‍🍳 <strong>Author:</strong> {recipe["author"]}</p>' if recipe['author'] else ""
    description_html = f'<p style="margin: 15px 0; font-style: italic; color: #555;">{recipe["description"]}</p>' if recipe['description'] else ""
    
    # 时间网格
    time_items = []
    if recipe['prep_time']: time_items.append(f'<div style="text-align: center; padding: 10px;"><div style="font-size: 11px; color: #666; text-transform: uppercase;">Prep Time</div><div style="font-size: 14px; font-weight: bold; color: #2c5f2d;">{recipe["prep_time"]}</div></div>')
    if recipe['cook_time']: time_items.append(f'<div style="text-align: center; padding: 10px;"><div style="font-size: 11px; color: #666; text-transform: uppercase;">Cook Time</div><div style="font-size: 14px; font-weight: bold; color: #2c5f2d;">{recipe["cook_time"]}</div></div>')
    if recipe['total_time']: time_items.append(f'<div style="text-align: center; padding: 10px;"><div style="font-size: 11px; color: #666; text-transform: uppercase;">Total Time</div><div style="font-size: 14px; font-weight: bold; color: #2c5f2d;">{recipe["total_time"]}</div></div>')
    time_info_html = f'<div style="display: flex; width: 100%; margin: 20px 0; border-top: 2px solid #97bc62; border-bottom: 2px solid #97bc62; padding: 15px 0;">' + "".join(time_items) + '</div>' if time_items else ""

    # 元数据网格 (包含 Servings, Cuisine, Category)
    meta_items = []
    if recipe['servings']: meta_items.append(f'<div style="text-align: center; padding: 8px;"><div style="font-size: 11px; color: #666; text-transform: uppercase;">Servings</div><div style="font-size: 14px; color: #2c5f2d; font-weight: bold;">{recipe["servings"]}</div></div>')
    if recipe['category']: meta_items.append(f'<div style="text-align: center; padding: 8px;"><div style="font-size: 11px; color: #666; text-transform: uppercase;">Course</div><div style="font-size: 14px; color: #2c5f2d;">{recipe["category"]}</div></div>')
    if recipe['cuisine']: meta_items.append(f'<div style="text-align: center; padding: 8px;"><div style="font-size: 11px; color: #666; text-transform: uppercase;">Cuisine</div><div style="font-size: 14px; color: #2c5f2d;">{recipe["cuisine"]}</div></div>')
    meta_html = f'<div style="display: flex; width: 100%; margin: 15px 0; background: #f9f9f9; border-radius: 8px;">' + "".join(meta_items) + '</div>' if meta_items else ""

    # 营养信息表格
    nutrition = recipe.get('nutrition', {})
    nutrition_html = ""
    if any(nutrition.values()):
        nutrition_items = []
        for key, label, unit in [('calories', 'Calories', 'kcal'), ('protein', 'Protein', 'g'), ('fat', 'Total Fat', 'g'), 
                                 ('saturated_fat', 'Saturated Fat', 'g'), ('carbohydrates', 'Carbohydrates', 'g'), 
                                 ('fiber', 'Fiber', 'g'), ('sugar', 'Sugar', 'g'), ('sodium', 'Sodium', 'mg'), ('cholesterol', 'Cholesterol', 'mg')]:
            value = nutrition.get(key, '')
            if value:
                if unit and not value.endswith(unit): value = f"{value} {unit}"
                nutrition_items.append(f'<tr><td style="padding: 6px 10px; border-bottom: 1px solid #eee;">{label}</td><td style="padding: 6px 10px; border-bottom: 1px solid #eee; text-align: right; font-weight: bold; color: #2c5f2d;">{value}</td></tr>')
        if nutrition_items:
            nutrition_html = f'''<div style="margin: 25px 0; border: 2px solid #97bc62; border-radius: 8px; overflow: hidden;">
                <h2 style="margin: 0; padding: 12px 15px; background: #97bc62; color: white; font-size: 16px; border: none;">📊 Nutrition Facts</h2>
                <table style="width: 100%; border-collapse: collapse; margin: 0;">{"".join(nutrition_items)}</table></div>'''

    image_html = f'<div style="text-align: center; margin: 20px 0;"><img src="{recipe["image"]}" style="max-width: 100%; height: auto; max-height: 400px; border-radius: 8px;" /></div>' if recipe['image'] else ""

    return f"""<html><head><meta charset="UTF-8"><style>
        @page {{ size: A4; margin: 1.5cm; }}
        body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; }}
        h1 {{ color: #2c5f2d; font-size: 28px; border-bottom: 3px solid #97bc62; padding-bottom: 12px; text-align: center; margin: 0 0 15px 0; }}
        h2 {{ color: #4a7c4b; font-size: 18px; margin: 25px 0 15px 0; padding: 10px 15px; background: #f0f7f0; border-left: 5px solid #97bc62; border-radius: 4px; }}
        ul {{ background: #fafafa; padding: 15px 15px 15px 35px; margin: 15px 0; border-left: 4px solid #97bc62; border-radius: 4px; }}
        li {{ margin-bottom: 10px; line-height: 1.5; }}
        ol li {{ margin-bottom: 15px; line-height: 1.5; }}
        .footer {{ margin-top: 40px; font-size: 11px; color: #999; text-align: center; border-top: 1px solid #ddd; padding-top: 15px; }}
    </style></head><body>
        <h1>{recipe['title']}</h1>
        {author_html}{description_html}{image_html}{time_info_html}{meta_html}{nutrition_html}
        <h2>🥗 Ingredients</h2><ul>{ingredients_html}</ul>
        <h2>‍🍳 Instructions</h2><ol>{instructions_html}</ol>
        <div class="footer"><p>Source: <a href="{recipe['url']}" style="color: #2c5f2d;">{recipe['url']}</a></p></div>
    </body></html>"""

def save_as_pdf(html_content, output_path):
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
    print(f"[*] 准备爬取 {len(target_urls)} 个菜谱...")
    
    for url in target_urls:
        recipe = extract_recipe_data(url, session)
        if recipe and recipe['ingredients']:
            html_content = generate_html(recipe)
            safe_filename = "".join([c for c in recipe['title'] if c.isalnum() or c in (' ', '-', '_')]).rstrip()
            pdf_path = os.path.join(OUTPUT_DIR, f"{safe_filename}.pdf")
            if not save_as_pdf(html_content, pdf_path):
                print(f"\n[✅] 成功导出 PDF: {pdf_path}")
            else:
                print(f"\n[!] PDF 生成失败")
        time.sleep(REQUEST_DELAY)
    print(f"\n[✅] 任务完成！文件保存在: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()