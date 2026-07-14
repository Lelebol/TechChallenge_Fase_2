"""Executa o notebook de ponta a ponta e grava as saidas nele (via kernel in-process).

Por que este script existe:
    Neste ambiente (Windows + Python 3.14) o kernel Jupyter padrao sobe, mas o handshake
    TCP/zmq entre kernel e cliente nao completa ("Kernel didn't respond"), o que impede rodar
    o notebook pelo Jupyter/nbconvert. A solucao usada aqui e o **kernel in-process** do
    ipykernel: ele roda no mesmo processo e troca mensagens por canais em memoria (sem TCP/zmq),
    sendo um shell Jupyter completo - produz tabelas (text/html), figuras (image/png via
    %matplotlib inline), stdout e erros, exatamente como um notebook normal.

Uso:
    python scripts/run_notebook.py notebooks/analise_wine_quality.ipynb

    O caminho do notebook pode ser relativo ou absoluto. O script muda o diretorio de trabalho
    para a pasta do notebook (para que os caminhos relativos e a deteccao da raiz do projeto
    funcionem), executa todas as celulas de codigo, embute as saidas e regrava o arquivo .ipynb.
    Encerra com codigo 2 se alguma celula gerar erro.
"""
import os
import sys
from pathlib import Path
from queue import Empty


def main():
    if len(sys.argv) < 2:
        print("Uso: python scripts/run_notebook.py <caminho_do_notebook.ipynb>")
        sys.exit(1)

    nb_path = Path(sys.argv[1]).resolve()
    if not nb_path.exists():
        print(f"Notebook nao encontrado: {nb_path}")
        sys.exit(1)

    os.chdir(nb_path.parent)  # cwd = notebooks/ -> caminhos relativos e find_project_root ok

    import nbformat
    from ipykernel.inprocess.manager import InProcessKernelManager

    nb = nbformat.read(nb_path, as_version=4)

    km = InProcessKernelManager()
    km.start_kernel()
    kc = km.client()
    kc.start_channels()

    counter = 0
    had_error = False
    first_error = None

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        counter += 1
        msg_id = kc.execute(cell.source, store_history=True)

        outputs = []
        exec_count = counter
        while True:
            try:
                msg = kc.get_iopub_msg(timeout=300)
            except Empty:
                break
            if msg.get("parent_header", {}).get("msg_id") not in (msg_id, None):
                continue
            mtype = msg["header"]["msg_type"]
            content = msg["content"]
            if mtype == "status":
                if content.get("execution_state") == "idle":
                    break
            elif mtype == "stream":
                outputs.append(nbformat.v4.new_output("stream", name=content["name"], text=content["text"]))
            elif mtype == "execute_result":
                exec_count = content.get("execution_count", counter)
                outputs.append(nbformat.v4.new_output(
                    "execute_result", data=content["data"],
                    metadata=content.get("metadata", {}), execution_count=exec_count))
            elif mtype == "display_data":
                outputs.append(nbformat.v4.new_output(
                    "display_data", data=content["data"], metadata=content.get("metadata", {})))
            elif mtype == "error":
                had_error = True
                if first_error is None:
                    first_error = (counter, content["ename"], content["evalue"])
                outputs.append(nbformat.v4.new_output(
                    "error", ename=content["ename"], evalue=content["evalue"],
                    traceback=content["traceback"]))

        try:
            reply = kc.get_shell_msg(timeout=10)
            ec = reply["content"].get("execution_count")
            if ec:
                exec_count = ec
        except Empty:
            pass

        cell.execution_count = exec_count
        cell.outputs = outputs
        if had_error:
            break

    try:
        kc.stop_channels()
        km.shutdown_kernel()
    except Exception:
        pass

    nbformat.write(nb, nb_path)

    if had_error:
        idx, ename, evalue = first_error
        print(f"ERRO na celula de codigo #{idx}: {ename}: {evalue}")
        sys.exit(2)
    print(f"OK: {counter} celulas de codigo executadas sem erro (kernel in-process).")


if __name__ == "__main__":
    main()
