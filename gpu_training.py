import subprocess
import requests
from fastapi import FastAPI
from config import CONTRACT, ACCOUNT, PRIVATE_KEY, PINATA_API_KEY, PINATA_SECRET_API_KEY
from web3 import Web3

app = FastAPI()

PINATA_UPLOAD_URL = "https://api.pinata.cloud/pinning/pinFileToIPFS"

def upload_to_pinata(file_path):
    """Uploads file to Pinata (IPFS) and returns the CID."""
    headers = {
        "pinata_api_key": PINATA_API_KEY,
        "pinata_secret_api_key": PINATA_SECRET_API_KEY
    }
    with open(file_path, "rb") as file:
        response = requests.post(PINATA_UPLOAD_URL, headers=headers, files={"file": file})
    return response.json()["IpfsHash"]

@app.post("/train_model/{job_id}")
def train_model(job_id: int):
    """Fetch dataset, train model with DeepSpeed, upload results to Pinata, and update blockchain."""
    
    # Fetch job details from smart contract
    job = CONTRACT.functions.jobs(job_id).call()
    model_cid = job[2]
    dataset_cid = job[3]

    # Construct IPFS URLs
    model_url = f"https://gateway.pinata.cloud/ipfs/{model_cid}"
    dataset_url = f"https://gateway.pinata.cloud/ipfs/{dataset_cid}"

    # Download model and dataset from IPFS
    model_path = "model.pth"
    dataset_path = "dataset.csv"

    with open(model_path, "wb") as model_file:
        model_file.write(requests.get(model_url).content)

    with open(dataset_path, "wb") as dataset_file:
        dataset_file.write(requests.get(dataset_url).content)

    # Run DeepSpeed training instead of Horovod
    command = "deepspeed --num_gpus=4 train.py"
    subprocess.run(command, shell=True)

    # Upload trained model to Pinata (IPFS)
    trained_model_cid = upload_to_pinata("trained_model.pth")

    # Notify blockchain
    txn = CONTRACT.functions.completeJob(job_id, trained_model_cid).build_transaction({
        "from": ACCOUNT,
        "gas": 500000,
        "gasPrice": Web3.to_wei("10", "gwei"),
        "nonce": Web3.eth.get_transaction_count(ACCOUNT)
    })

    signed_txn = Web3.eth.account.sign_transaction(txn, PRIVATE_KEY)
    tx_hash = Web3.eth.send_raw_transaction(signed_txn.rawTransaction)

    return {"trained_model_cid": trained_model_cid, "tx_hash": tx_hash.hex()}
