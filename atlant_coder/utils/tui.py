def wait_user_enter(asks: str):
    cmd = input(
        f"{asks}\n\nНачните вводить сюда ваш ответ, как закончите дважды нажмите enter.\n> "
    )
    answer = f"{cmd}\n"

    exit_proof = 0
    exit_status = 0
    while not exit_status:
        if not cmd:
            exit_proof += 1

        if exit_proof == 2:
            exit_status = 1
            return answer

        cmd = input("> ")
        answer += f"{cmd}\n"

    return answer
