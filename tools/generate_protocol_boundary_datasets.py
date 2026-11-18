#!/usr/bin/env python3
"""Generate synthetic boundary-case datasets for protocol consistency evaluation.

Produces three JSONL files under the protocol_constrained dataset directory:
    - boundary_regex.jsonl       (~20 cases: multi-byte, fixed-length, mixed patterns)
    - boundary_grammar.jsonl     (~20 cases: SQL, arithmetic, YAML, Python, CSV, JSON EBNF)
    - boundary_adversarial.jsonl (~30 cases: tag collision, escaping, tool_choice, nesting)

Each JSONL line follows the existing protocol dataset format:
    {"id": "...", "question": "...", "max_out_len": N,
     "extra_params": {...}, "ground_truth": {...}}

Usage:
    python tools/generate_protocol_boundary_datasets.py
    python tools/generate_protocol_boundary_datasets.py --output-dir <path>
"""

import argparse
import json
import os


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _regex_case(id_, question, pattern, max_out_len=2048):
    """Build a regex-constrained case."""
    return {
        "id": id_,
        "question": question,
        "max_out_len": max_out_len,
        "extra_params": {"guided_regex": pattern},
        "ground_truth": {"expected_finish_reason": "stop", "regex": pattern},
    }


def _grammar_case(id_, question, grammar, answer_regex, max_out_len=2048):
    """Build an EBNF grammar-constrained case."""
    return {
        "id": id_,
        "question": question,
        "max_out_len": max_out_len,
        "extra_params": {"guided_grammar": grammar},
        "ground_truth": {
            "expected_finish_reason": "stop",
            "grammar": True,
            "answer_regex": answer_regex,
        },
    }


def _schema_case(id_, question, schema, max_out_len=4096,
                 ground_truth_extra=None):
    """Build a JSON Schema-constrained case."""
    gt = {"expected_finish_reason": "stop", "schema": schema}
    if ground_truth_extra:
        gt.update(ground_truth_extra)
    return {
        "id": id_,
        "question": question,
        "max_out_len": max_out_len,
        "extra_params": {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": id_, "strict": True, "schema": schema},
            },
        },
        "ground_truth": gt,
    }


def _tool_case(id_, question, tools, tool_choice, ground_truth, max_out_len=2048):
    """Build a tool-call case."""
    return {
        "id": id_,
        "question": question,
        "max_out_len": max_out_len,
        "extra_params": {"tools": tools, "tool_choice": tool_choice},
        "ground_truth": ground_truth,
    }


# ---------------------------------------------------------------------------
# Regex boundary cases (~20)
# ---------------------------------------------------------------------------

def generate_regex_cases():
    """Multi-byte UTF-8, fixed-length IDs, mixed separators, network, domain."""
    cases = []

    # --- Multi-byte UTF-8 ---
    cases.append(_regex_case(
        "boundary_regex_001",
        "生成一个5到10个汉字的中文字符串，不要包含任何标点符号或空格。",
        r"[\u4e00-\u9fa5]{5,10}",
    ))
    cases.append(_regex_case(
        "boundary_regex_002",
        "生成一个中国大陆车牌号，格式为：省份简称（汉字）+大写字母+5位字母数字，例如京A12345。",
        r"[\u4e00-\u9fa5][A-Z][A-Z0-9]{5}",
    ))
    cases.append(_regex_case(
        "boundary_regex_003",
        "生成一个10到20个字符的中文句子，可以包含中文标点符号（，。！？、）。",
        r"[\u4e00-\u9fa5，。！？、]{10,20}",
    ))

    # --- Fixed-length ID patterns ---
    cases.append(_regex_case(
        "boundary_regex_004",
        "生成一个18位中国大陆身份证号码，前17位为数字，最后一位可以是数字或字母X。",
        r"\d{17}[\dXx]",
    ))
    cases.append(_regex_case(
        "boundary_regex_005",
        "生成一个15位旧版中国大陆身份证号码，全部为数字。",
        r"\d{15}",
    ))
    cases.append(_regex_case(
        "boundary_regex_006",
        "生成一个中国护照号码，格式为字母E或G开头后跟8位数字。",
        r"[EGeg]\d{8}",
    ))
    cases.append(_regex_case(
        "boundary_regex_007",
        "生成一个6位中国邮政编码。",
        r"[0-9]{6}",
    ))

    # --- Mixed alphanumeric with separators ---
    cases.append(_regex_case(
        "boundary_regex_008",
        "生成一个订单编码，格式为ORD-后跟4位数字-后跟2个大写字母，例如ORD-1234-AB。",
        r"ORD-\d{4}-[A-Z]{2}",
    ))
    cases.append(_regex_case(
        "boundary_regex_009",
        "生成一个UUID，格式为8-4-4-4-12位十六进制数字，用连字符分隔。",
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
    ))
    cases.append(_regex_case(
        "boundary_regex_010",
        "生成一个MAC地址，格式为6组2位十六进制数，用冒号分隔，例如AA:BB:CC:DD:EE:FF。",
        r"([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}",
    ))

    # --- Network-related ---
    cases.append(_regex_case(
        "boundary_regex_011",
        "生成一个IPv6地址，格式为8组1到4位十六进制数，用冒号分隔。",
        r"([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}",
    ))
    cases.append(_regex_case(
        "boundary_regex_012",
        "生成一个带国家码的手机号码，格式为+86后跟11位数字（第一位非0）。",
        r"\+86[1-9]\d{9}",
    ))
    cases.append(_regex_case(
        "boundary_regex_013",
        "生成一个HTTP或HTTPS URL，包含域名和可选路径。",
        r"https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?",
    ))

    # --- Time and version formats ---
    cases.append(_regex_case(
        "boundary_regex_014",
        "生成一个24小时制时间戳，格式为HH:MM:SS。",
        r"([01]\d|2[0-3]):[0-5]\d:[0-5]\d",
    ))
    cases.append(_regex_case(
        "boundary_regex_015",
        "生成一个语义化版本号，格式为major.minor.patch，例如1.2.3。",
        r"\d+\.\d+\.\d+",
    ))

    # --- Domain-specific ---
    cases.append(_regex_case(
        "boundary_regex_016",
        "生成一个电子邮件地址。",
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    ))
    cases.append(_regex_case(
        "boundary_regex_017",
        "生成一个6位十六进制颜色代码，以#开头，例如#FF5733。",
        r"#[0-9a-fA-F]{6}",
    ))
    cases.append(_regex_case(
        "boundary_regex_018",
        "生成一个GPS坐标，格式为纬度,经度，保留4到6位小数，例如39.9042,116.4074。",
        r"-?\d+\.\d{4,6},-?\d+\.\d{4,6}",
    ))
    cases.append(_regex_case(
        "boundary_regex_019",
        "生成一个ISBN-13号码，格式为978或979开头，用连字符分5段。",
        r"97[89]-\d{1,5}-\d{1,7}-\d{1,7}-\d",
    ))
    cases.append(_regex_case(
        "boundary_regex_020",
        "生成一个16到19位的银行卡号，全部为数字。",
        r"\d{16,19}",
    ))

    return cases


# ---------------------------------------------------------------------------
# Grammar boundary cases (~20)
# ---------------------------------------------------------------------------

def generate_grammar_cases():
    """Six EBNF grammar sets: SQL, arithmetic, YAML, Python, CSV, JSON, assignment."""
    cases = []

    # --- SQL SELECT subset ---
    sql_grammar = (
        'root ::= select_stmt\n'
        'select_stmt ::= "SELECT " column_list " FROM " table_name where_clause?\n'
        'column_list ::= column_name (", " column_name)*\n'
        'column_name ::= [a-zA-Z_][a-zA-Z0-9_]*\n'
        'table_name ::= [a-zA-Z_][a-zA-Z0-9_]*\n'
        'where_clause ::= " WHERE " condition\n'
        'condition ::= column_name " = " value\n'
        'value ::= [0-9]+ | "\'" [a-zA-Z0-9 ]* "\'"'
    )
    cases.append(_grammar_case(
        "boundary_grammar_001",
        "写一个SQL查询语句，从users表中选择name列。",
        sql_grammar, r"SELECT name FROM users",
    ))
    cases.append(_grammar_case(
        "boundary_grammar_002",
        "写一个SQL查询语句，从orders表中选择id和amount列，条件是status = 'paid'。",
        sql_grammar, r"SELECT id, amount FROM orders WHERE status = 'paid'",
    ))
    cases.append(_grammar_case(
        "boundary_grammar_003",
        "写一个SQL查询语句，从products表中选择name、price和category列，条件是price = 100。",
        sql_grammar, r"SELECT name, price, category FROM products WHERE price = 100",
    ))

    # --- Simple arithmetic expression ---
    arith_grammar = (
        'root ::= expr\n'
        'expr ::= num op num\n'
        'num ::= [0-9]+\n'
        'op ::= " + " | " - " | " * " | " / "'
    )
    cases.append(_grammar_case(
        "boundary_grammar_004",
        "生成一个加法算术表达式，包含两个数字。",
        arith_grammar, r"[0-9]+ \+ [0-9]+",
    ))
    cases.append(_grammar_case(
        "boundary_grammar_005",
        "生成一个乘法算术表达式，包含两个数字。",
        arith_grammar, r"[0-9]+ \* [0-9]+",
    ))
    cases.append(_grammar_case(
        "boundary_grammar_006",
        "生成一个减法算术表达式，包含两个两位数。",
        arith_grammar, r"[0-9]+ \- [0-9]+",
    ))

    # --- YAML key-value ---
    yaml_grammar = (
        'root ::= (key ": " value "\n")+\n'
        'key ::= [a-zA-Z_][a-zA-Z0-9_]*\n'
        'value ::= [a-zA-Z0-9 .-]+'
    )
    cases.append(_grammar_case(
        "boundary_grammar_007",
        "生成一个YAML配置，包含name和version两个配置项。",
        yaml_grammar, r"name: .+\nversion: .+",
    ))
    cases.append(_grammar_case(
        "boundary_grammar_008",
        "生成一个YAML配置，包含host、port和debug三个配置项。",
        yaml_grammar, r"host: .+\nport: .+\ndebug: .+",
    ))
    cases.append(_grammar_case(
        "boundary_grammar_009",
        "生成一个YAML配置，包含database、user和password三个配置项。",
        yaml_grammar, r"database: .+\nuser: .+\npassword: .+",
    ))

    # --- Python function signature ---
    py_grammar = (
        'root ::= "def " name "(" params ")" ":"\n'
        'name ::= [a-zA-Z_][a-zA-Z0-9_]*\n'
        'params ::= param (", " param)* | ""\n'
        'param ::= name ": " type\n'
        'type ::= "int" | "str" | "float" | "bool"'
    )
    cases.append(_grammar_case(
        "boundary_grammar_010",
        "写一个Python函数签名，函数名为greet，参数name类型为str。",
        py_grammar, r"def greet\(name: str\):",
    ))
    cases.append(_grammar_case(
        "boundary_grammar_011",
        "写一个Python函数签名，函数名为calculate，参数a类型为int，参数b类型为float。",
        py_grammar, r"def calculate\(a: int, b: float\):",
    ))
    cases.append(_grammar_case(
        "boundary_grammar_012",
        "写一个Python函数签名，函数名为process，参数data类型为str，count类型为int，verbose类型为bool。",
        py_grammar, r"def process\(data: str, count: int, verbose: bool\):",
    ))

    # --- CSV row ---
    csv_grammar = (
        'root ::= field ("," field)* "\n"\n'
        'field ::= [^,\n"]* | "\\"" [^"]*\\""'
    )
    cases.append(_grammar_case(
        "boundary_grammar_013",
        "生成一行CSV数据，包含姓名、年龄和城市三个字段。",
        csv_grammar, r"[^,\n]+,[^,\n]+,[^,\n]+",
    ))
    cases.append(_grammar_case(
        "boundary_grammar_014",
        "生成一行CSV数据，包含产品名、价格和库存三个字段。",
        csv_grammar, r"[^,\n]+,[^,\n]+,[^,\n]+",
    ))
    cases.append(_grammar_case(
        "boundary_grammar_015",
        "生成一行CSV数据，包含书名、作者和出版年份三个字段。",
        csv_grammar, r"[^,\n]+,[^,\n]+,[^,\n]+",
    ))

    # --- JSON object ---
    json_grammar = (
        'root ::= "{" key_value (", " key_value)* "}"\n'
        'key_value ::= key ": " value\n'
        'key ::= "\\"" [a-zA-Z_][a-zA-Z0-9_]* "\\""\n'
        'value ::= "\\"" [^"]*\\"" | [0-9]+ | "true" | "false"'
    )
    cases.append(_grammar_case(
        "boundary_grammar_016",
        "生成一个JSON对象，包含name（字符串）和age（数字）两个字段。",
        json_grammar, r'\{"name": "[^"]*", "age": [0-9]+\}',
    ))
    cases.append(_grammar_case(
        "boundary_grammar_017",
        "生成一个JSON对象，包含city（字符串）和population（数字）两个字段。",
        json_grammar, r'\{"city": "[^"]*", "population": [0-9]+\}',
    ))
    cases.append(_grammar_case(
        "boundary_grammar_018",
        "生成一个JSON对象，包含active（布尔值）和count（数字）两个字段。",
        json_grammar, r'\{"active": (true|false), "count": [0-9]+\}',
    ))

    # --- Assignment with precedence ---
    assign_grammar = (
        'root ::= assignment\n'
        'assignment ::= var " = " expr\n'
        'var ::= [a-zA-Z_][a-zA-Z0-9_]*\n'
        'expr ::= term ((" + " | " - ") term)*\n'
        'term ::= factor ((" * " | " / ") factor)*\n'
        'factor ::= [0-9]+ | var | "(" expr ")"'
    )
    cases.append(_grammar_case(
        "boundary_grammar_019",
        "生成一个变量赋值语句，将一个算术表达式的结果赋给变量x。",
        assign_grammar, r"x = .+",
    ))
    cases.append(_grammar_case(
        "boundary_grammar_020",
        "生成一个变量赋值语句，将一个包含括号的算术表达式的结果赋给变量result。",
        assign_grammar, r"result = .+",
    ))

    return cases


# ---------------------------------------------------------------------------
# Adversarial boundary cases (~30)
# ---------------------------------------------------------------------------

def _nest_obj(key, inner):
    """Object schema with one required property containing inner schema."""
    return {"type": "object", "properties": {key: inner}, "required": [key]}


def _nest_arr(key, inner):
    """Object schema with one required property that is an array of inner."""
    return {"type": "object", "properties": {key: {"type": "array", "items": inner}}, "required": [key]}


def generate_adversarial_cases():
    """Tag collision, escaping, tool_choice, minimal content, deep nesting."""
    cases = []

    # === 1. Tag collision (5) — reasoning tag strings in prompt ===
    tag_schema = {
        "type": "object",
        "properties": {
            "tag_name": {"type": "string"},
            "description": {"type": "string"},
            "example": {"type": "string"},
        },
        "required": ["tag_name", "description", "example"],
    }
    cases.append(_schema_case(
        "boundary_adv_tag_001",
        "请解释XML中<dim_think>元素的作用，以JSON格式输出，包含tag_name、description和example三个字段。",
        tag_schema))
    cases.append(_schema_case(
        "boundary_adv_tag_002",
        "请用JSON格式说明<dim_think>和</dim_think>标签在推理模型中的用途，包含tag_name、description和example字段。",
        tag_schema))
    cases.append(_schema_case(
        "boundary_adv_tag_003",
        "写一段关于HTML自定义标签<dim_think>的说明，以JSON格式输出，包含tag_name、description和example字段。",
        tag_schema))
    cases.append(_schema_case(
        "boundary_adv_tag_004",
        "请解释Markdown中如何展示<dim_think>标签的原始文本，以JSON格式输出，包含tag_name、description和example字段。",
        tag_schema))
    cases.append(_schema_case(
        "boundary_adv_tag_005",
        "请说明在XML文档中<dim_think>标签的转义方法，以JSON格式输出，包含tag_name、description和example字段。",
        tag_schema))

    # === 2. Special character escaping (8) ===
    cases.append(_schema_case(
        "boundary_adv_escape_001",
        '生成一个JSON对象，包含message字段，值须含双引号字符，例如：He said "Hello"。',
        {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}))
    cases.append(_schema_case(
        "boundary_adv_escape_002",
        "生成一个JSON对象，包含path字段，值须含反斜杠字符，例如：C:\\\\Users\\\\test。",
        {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}))
    cases.append(_schema_case(
        "boundary_adv_escape_003",
        "生成一个JSON对象，包含text字段，值须含换行符。",
        {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}))
    cases.append(_schema_case(
        "boundary_adv_escape_004",
        "生成一个JSON对象，包含含有Emoji表情\U0001F468\u200D\U0001F469\u200D\U0001F467\u200D\U0001F466的message字段。",
        {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}))
    cases.append(_schema_case(
        "boundary_adv_escape_005",
        '生成一个JSON对象，包含content字段，值须同时含双引号(")、反斜杠(\\)和换行符。',
        {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}))
    cases.append(_schema_case(
        "boundary_adv_escape_006",
        "生成一个JSON对象，包含text字段，值须同时含中文、日文(こんにちは)和韩文(안녕하세요)。",
        {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}))
    cases.append(_schema_case(
        "boundary_adv_escape_007",
        "生成一个JSON对象，name字段值为空字符串，id字段值为数字1。",
        {"type": "object", "properties": {"name": {"type": "string"}, "id": {"type": "number"}}, "required": ["name", "id"]}))
    cases.append(_schema_case(
        "boundary_adv_escape_008",
        "生成一个JSON对象，description字段值须为超过100字的中文描述。",
        {"type": "object", "properties": {"description": {"type": "string"}}, "required": ["description"]},
        max_out_len=8192))

    # === 3. tool_choice forced mode (9) ===
    weather_tool = [{"type": "function", "function": {"name": "get_weather", "description": "获取指定城市的天气信息", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "城市名称"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}]
    calc_tool = [{"type": "function", "function": {"name": "calculate", "description": "执行数学计算", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式"}}, "required": ["expression"]}}}]
    multi_tool = weather_tool + calc_tool

    # tool_choice="none" (3) — must NOT call any tool
    cases.append(_tool_case("boundary_adv_tool_none_001",
        "北京今天天气怎么样？", weather_tool, "none",
        {"expected_finish_reason": "stop"}))
    cases.append(_tool_case("boundary_adv_tool_none_002",
        "帮我计算 123 乘以 456。", calc_tool, "none",
        {"expected_finish_reason": "stop"}))
    cases.append(_tool_case("boundary_adv_tool_none_003",
        "查一下上海的天气，然后算 99 + 1。", multi_tool, "none",
        {"expected_finish_reason": "stop"}))

    # tool_choice="required" (3) — MUST call some tool
    cases.append(_tool_case("boundary_adv_tool_req_001",
        "今天天气如何？", weather_tool, "required",
        {"expected_finish_reason": "tool_calls", "tool_calls": [{"function": {"name": "get_weather"}}]}))
    cases.append(_tool_case("boundary_adv_tool_req_002",
        "帮我算点东西。", calc_tool, "required",
        {"expected_finish_reason": "tool_calls", "tool_calls": [{"function": {"name": "calculate"}}]}))
    cases.append(_tool_case("boundary_adv_tool_req_003",
        "我需要一些信息。", multi_tool, "required",
        {"expected_finish_reason": "tool_calls"}))

    # tool_choice=specific function (3)
    cases.append(_tool_case("boundary_adv_tool_spec_001",
        "帮我查一下北京的天气。", multi_tool,
        {"type": "function", "function": {"name": "get_weather"}},
        {"expected_finish_reason": "tool_calls", "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "北京"}}}]}))
    cases.append(_tool_case("boundary_adv_tool_spec_002",
        "帮我计算 25 * 4。", multi_tool,
        {"type": "function", "function": {"name": "calculate"}},
        {"expected_finish_reason": "tool_calls", "tool_calls": [{"function": {"name": "calculate", "arguments": {"expression": "25*4"}}}]}))
    cases.append(_tool_case("boundary_adv_tool_spec_003",
        "今天北京天气怎么样？顺便算 10 + 20。", multi_tool,
        {"type": "function", "function": {"name": "get_weather"}},
        {"expected_finish_reason": "tool_calls", "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "北京"}}}]}))

    # === 4. Empty/minimal content (4) ===
    cases.append(_schema_case("boundary_adv_minimal_001",
        "请输出一个空JSON对象{}。",
        {"type": "object"}))
    cases.append({"id": "boundary_adv_minimal_002",
        "question": "请只输出数字42，不要输出任何其他内容。",
        "max_out_len": 128, "extra_params": {},
        "ground_truth": {"expected_finish_reason": "stop", "regex": "42"}})
    cases.append({"id": "boundary_adv_minimal_003",
        "question": "请只输出单词yes，不要输出任何其他内容。",
        "max_out_len": 128, "extra_params": {},
        "ground_truth": {"expected_finish_reason": "stop", "regex": "yes"}})
    cases.append({"id": "boundary_adv_minimal_004",
        "question": "请只输出布尔值true，不要输出任何其他内容。",
        "max_out_len": 128, "extra_params": {},
        "ground_truth": {"expected_finish_reason": "stop", "regex": "true"}})

    # === 5. Deep nested schema (4) ===
    _leaf_person = {"type": "object", "properties": {"name": {"type": "string"}, "role": {"type": "string"}}, "required": ["name", "role"]}
    _leaf_planet = {"type": "object", "properties": {"name": {"type": "string"}, "mass": {"type": "number"}}, "required": ["name", "mass"]}
    _leaf_course = {"type": "object", "properties": {"name": {"type": "string"}, "credits": {"type": "integer"}}, "required": ["name", "credits"]}
    _leaf_street = {"type": "object", "properties": {"name": {"type": "string"}, "type": {"type": "string", "enum": ["residential", "commercial"]}}, "required": ["name", "type"]}

    # 5-level object nesting
    cases.append(_schema_case("boundary_adv_deep_001",
        "生成一个5层嵌套的JSON：公司→部门→团队→成员（含name和role）。",
        _nest_obj("company", _nest_obj("department", _nest_obj("team", _nest_obj("member", _leaf_person))))))
    # 6-level with arrays
    cases.append(_schema_case("boundary_adv_deep_002",
        "生成一个6层嵌套的JSON：宇宙→星系列表→恒星系统列表→行星列表（含name和mass）。",
        _nest_obj("universe", _nest_arr("galaxies", _nest_arr("star_systems", _nest_arr("planets", _leaf_planet)))),
        max_out_len=8192))
    # 7-level mixed
    cases.append(_schema_case("boundary_adv_deep_003",
        "生成一个7层嵌套的JSON：学校→学院→系→专业→课程列表（含name和credits）。",
        _nest_obj("school", _nest_obj("college", _nest_obj("department", _nest_obj("major", _nest_arr("courses", _leaf_course))))),
        max_out_len=8192))
    # 8-level with enum
    cases.append(_schema_case("boundary_adv_deep_004",
        "生成一个8层嵌套的JSON：国家→省份列表→城市列表→区县列表→街道列表（含name和type，type为residential或commercial）。",
        _nest_obj("country", _nest_arr("provinces", _nest_arr("cities", _nest_arr("districts", _nest_arr("streets", _leaf_street))))),
        max_out_len=8192))

    return cases


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_jsonl(cases, filepath):
    """Write a list of case dicts to a JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(cases)} cases to {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate boundary-case datasets for protocol evaluation")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "xtask", "benchmarks", "aisbench", "datasets", "protocol_constrained"),
        help="Output directory for generated JSONL files")
    args = parser.parse_args()

    print("Generating boundary regex cases...")
    write_jsonl(generate_regex_cases(),
                os.path.join(args.output_dir, "boundary_regex.jsonl"))
    print("Generating boundary grammar cases...")
    write_jsonl(generate_grammar_cases(),
                os.path.join(args.output_dir, "boundary_grammar.jsonl"))
    print("Generating adversarial boundary cases...")
    write_jsonl(generate_adversarial_cases(),
                os.path.join(args.output_dir, "boundary_adversarial.jsonl"))

    total = 20 + 20 + 30
    print(f"\nDone: {total} cases generated in {args.output_dir}")


if __name__ == "__main__":
    main()