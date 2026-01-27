from rag import ask_pdf

while True:
    q = input("\nAsk a question (type exit to quit): ")
    if q.lower() == "exit":
        break

    ans = ask_pdf(q)
    print("\nAnswer:\n", ans)
