from web3 import Web3
from fastapi import FastAPI
from config import GPU_REGISTRATION_CONTRACT, ACCOUNT, PRIVATE_KEY,INFURA_URL

app = FastAPI()
web3 = Web3(Web3.HTTPProvider(INFURA_URL))


@app.post("/register_gpu")
def register_gpu(hardware_details: str):
    print("hardware_details")
    """Registers a GPU node on-chain."""
    txn = GPU_REGISTRATION_CONTRACT.functions.registerNode(hardware_details).build_transaction({
        "from": ACCOUNT,
        "gas": 1000000,
        "gasPrice": web3.to_wei("30", "gwei"),
        "nonce": web3.eth.get_transaction_count(ACCOUNT)
    })
    
    signed_txn = web3.eth.account.sign_transaction(txn, PRIVATE_KEY)
    tx_hash = web3.eth.send_raw_transaction(signed_txn.raw_transaction)
    
    return {"tx_hash": tx_hash.hex()}


@app.post("/mark_gpu_available")
def mark_gpu_available(node_address: str):
    """Marks a GPU as available again."""
    txn = GPU_REGISTRATION_CONTRACT.functions.markNodeAvailable(node_address).build_transaction({
        "from": ACCOUNT,
        "gas": 1000000,
        "gasPrice": web3.to_wei("30", "gwei"),
        "nonce": web3.eth.get_transaction_count(ACCOUNT)
    })
    
    signed_txn = web3.eth.account.sign_transaction(txn, PRIVATE_KEY)
    tx_hash = web3.eth.send_raw_transaction(signed_txn.raw_transaction)
    
    return {"tx_hash": tx_hash.hex()}
