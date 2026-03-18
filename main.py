from pipeline.ai_tutor import AITutor

def main():
    tutor = AITutor()
    print("\n" + "="*60)
    print("🎓 GIA SƯ AI")
    print("="*60)

    while True:
        q = input("\n❓ Câu hỏi: ").strip()
        if q.lower() == 'exit': break
        if not q: continue

        a = input("👨‍🎓 Học sinh: ").strip()
        if a.lower() == 'exit': break
        if not a: continue

        keyword = tutor.extract_keywords(q)
        truth = tutor.get_ground_truth(keyword)
        
        if not truth:
            print("⚠️ Không tìm thấy bài học chuẩn trong hệ thống.")
            continue

        analysis = tutor.verify_and_analyze(q, a, truth['content'])
        misconcept = None

        if analysis['is_correct']:
            print("✅ Đánh giá: ĐÚNG.")
        elif not analysis['is_meaningful']:
            print("🚫 Đánh giá: Câu trả lời vô nghĩa/lạc đề (Bỏ qua lưu trữ).")
        else:
            print(f"❌ Đánh giá: SAI. Phân tích lỗi: {analysis['suggested_name']}")
            misconcept = tutor.retrieve_misconception(analysis['logical_break'])
            
            if misconcept:
                print(f"🎯 Lỗi này ĐÃ CÓ trong Graph (Độ tự tin: {misconcept['score']:.2f})")
            else:
                print("🧠 Phát hiện lỗi MỚI. Đang tự động nạp vào Knowledge Graph...")
                tutor.learn_new_misconception(truth['concept'], analysis)
                print("✅ Đã cập nhật database thành công!")

        print("\n💬 [GIA SƯ AI]:")
        feedback = tutor.generate_feedback(q, a, analysis, truth, misconcept)
        print(feedback)
        print("-" * 50)

    tutor.close()

if __name__ == "__main__":
    main()