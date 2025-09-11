#!/usr/bin/env python3
"""
GitHub部署脚本 - 将债务预测系统上传到GitHub
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description=""):
    """运行命令并处理错误"""
    print(f"\n🔄 {description}")
    print(f"   命令: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ 成功")
            if result.stdout.strip():
                print(f"   输出: {result.stdout.strip()}")
            return True
        else:
            print(f"   ❌ 失败")
            if result.stderr.strip():
                print(f"   错误: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"   ❌ 异常: {str(e)}")
        return False

def check_git_status():
    """检查Git状态"""
    print("\n📊 检查Git仓库状态...")
    
    # 检查是否在Git仓库中
    if not run_command("git status", "检查Git状态"):
        print("❌ 当前目录不是Git仓库，需要初始化")
        return False
    
    # 检查远程仓库
    result = subprocess.run("git remote -v", shell=True, capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        print("✅ 发现远程仓库:")
        print(f"   {result.stdout.strip()}")
        return True
    else:
        print("⚠️  未发现远程仓库")
        return False

def setup_git_repository():
    """设置Git仓库"""
    print("\n🔧 设置Git仓库...")
    
    # 初始化Git仓库（如果需要）
    if not os.path.exists('.git'):
        if not run_command("git init", "初始化Git仓库"):
            return False
    
    # 添加远程仓库
    github_url = "https://github.com/xiaobxj/sarima-method.git"
    
    # 检查是否已有origin
    result = subprocess.run("git remote get-url origin", shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        if not run_command(f"git remote add origin {github_url}", "添加远程仓库"):
            return False
    else:
        print(f"✅ 远程仓库已存在: {result.stdout.strip()}")
        # 更新远程仓库URL（如果不同）
        if github_url not in result.stdout:
            run_command(f"git remote set-url origin {github_url}", "更新远程仓库URL")
    
    return True

def prepare_files():
    """准备文件用于提交"""
    print("\n📁 准备文件...")
    
    # 检查重要文件是否存在
    important_files = [
        'README.md',
        'requirements.txt', 
        'LICENSE',
        '.gitignore',
        'src/debt_prediction/main.py',
        'src/debt_prediction/config.py',
        'src/debt_prediction/tga_simulator.py'
    ]
    
    missing_files = []
    for file in important_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  缺少重要文件: {', '.join(missing_files)}")
        return False
    
    print("✅ 所有重要文件都存在")
    return True

def commit_and_push():
    """提交并推送到GitHub"""
    print("\n📤 提交并推送到GitHub...")
    
    # 添加所有文件
    if not run_command("git add .", "添加所有文件"):
        return False
    
    # 检查是否有变更
    result = subprocess.run("git diff --cached --name-only", shell=True, capture_output=True, text=True)
    if not result.stdout.strip():
        print("ℹ️  没有发现变更，跳过提交")
        return True
    
    print(f"📝 发现变更文件:")
    for file in result.stdout.strip().split('\n'):
        print(f"   - {file}")
    
    # 提交变更
    commit_message = "feat: 完整的美国国债债务上限分析与预测系统\n\n- 实现TGA余额模拟器和X-Date预测\n- 集成20年历史现金流数据(2005-2025)\n- 支持债务到期日历和利息支付追踪\n- 提供180天滚动预测和风险评估\n- 包含完整的可视化和分析报告"
    
    if not run_command(f'git commit -m "{commit_message}"', "提交变更"):
        return False
    
    # 推送到GitHub (强制推送以替换现有内容)
    print("\n⚠️  准备强制推送以替换GitHub上的现有内容...")
    response = input("是否继续？这将完全替换GitHub上的现有项目 (y/N): ")
    
    if response.lower() in ['y', 'yes', '是']:
        if not run_command("git push -f origin main", "强制推送到GitHub"):
            # 尝试推送到master分支
            if not run_command("git push -f origin master", "强制推送到GitHub (master分支)"):
                return False
        
        print("\n🎉 成功上传到GitHub!")
        print(f"   项目地址: https://github.com/xiaobxj/sarima-method")
        return True
    else:
        print("❌ 用户取消了推送操作")
        return False

def main():
    """主函数"""
    print("🚀 GitHub部署脚本 - 美国国债债务预测系统")
    print("=" * 60)
    
    # 确保在正确的目录
    if not os.path.exists('src/debt_prediction'):
        print("❌ 错误: 请在项目根目录运行此脚本")
        sys.exit(1)
    
    # 步骤1: 检查Git状态
    git_exists = check_git_status()
    
    # 步骤2: 设置Git仓库
    if not setup_git_repository():
        print("❌ Git仓库设置失败")
        sys.exit(1)
    
    # 步骤3: 准备文件
    if not prepare_files():
        print("❌ 文件准备失败")
        sys.exit(1)
    
    # 步骤4: 提交并推送
    if not commit_and_push():
        print("❌ 提交推送失败")
        sys.exit(1)
    
    print("\n✅ 部署完成!")
    print("\n📋 后续步骤:")
    print("   1. 访问 https://github.com/xiaobxj/sarima-method")
    print("   2. 检查README和项目结构")
    print("   3. 如需要，可以编辑项目描述和标签")
    print("   4. 考虑添加GitHub Actions进行CI/CD")

if __name__ == "__main__":
    main()
