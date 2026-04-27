import os
import sys
import subprocess

def git_update():
    print("🔄 Performing Hard Reset and Repository Update...")
    try:
        subprocess.run(["git", "fetch", "origin"], check=True)
        subprocess.run(["git", "reset", "--hard", "origin/main"], check=True)
        print("📦 Installing/Verifying requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"], check=True)
        print("✅ System is fully synchronized with GitHub!")
    except Exception as e:
        print(f"⚠️ Git update failed (ignore if running locally without git): {e}")

def main():
    print("="*60)
    print("🚀 FOREX AI SYSTEM - LAUNCH MANAGER")
    print("="*60)
    print("Select Execution Mode:")
    print("  [1] Supervised Live Trading (XGBoost + HMM)")
    print("  [2] Deep Reinforcement Learning (PPO Matrix Simulation)")
    print("="*60)

    # Allow CLI argument like: python run.py 2
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nEnter 1 or 2: ").strip()

    if choice not in ["1", "2"]:
        print("❌ Invalid selection. Exiting.")
        sys.exit(1)

    git_update()

    print("\n" + "="*60)
    if choice == "1":
        print("🚀 LAUNCHING COLAB/KAGGLE MASTER LOOP (MODE 1)")
        print("="*60)
        os.system(f"{sys.executable} notebooks/colab_master_loop.py")
    elif choice == "2":
        print("🧠 LAUNCHING RL MATRIX TRAINER (MODE 2)")
        print("="*60)
        os.system(f"{sys.executable} notebooks/train_rl_agent.py")

if __name__ == "__main__":
    main()
