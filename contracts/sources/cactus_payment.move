module cactus::payment {
    use std::signer;
    use aptos_framework::event;
    use std::vector;
    use std::bcs;

    /// Errors
    const EINVALID_TASK: u64 = 1;
    const EINVALID_ORCHESTRATOR: u64 = 2;
    const EINSUFFICIENT_BALANCE: u64 = 3;
    const EINVALID_STATUS: u64 = 4;
    const EINVALID_CLIENT: u64 = 5;
    const ENO_TASKS: u64 = 6;
    const EALREADY_REGISTERED: u64 = 7;
    const EINVALID_DELEGATION: u64 = 8;
    const ENOT_REGISTERED: u64 = 9;
    const EUNAUTHORIZED_SERVER: u64 = 10;
    const EALREADY_AUTHORIZED: u64 = 11;
    const EINVALID_SERVER: u64 = 12;

    /// Task status
    const STATUS_PENDING: u8 = 0;
    const STATUS_COMPLETED: u8 = 1;
    const STATUS_FAILED: u8 = 2;
    const STATUS_DISPUTED: u8 = 3;

    /// Task struct to store task information
    struct Task has key, store, copy, drop {
        client: address,              // Server address that created the task
        client_identifier: vector<u8>, // Identifier for the actual client
        orchestrator: address,
        compute_time: u64,
        gpu_resources: u64,
        model_type: vector<u8>,
        payment_amount: u64,
        result_hash: vector<u8>,
        status: u8
    }

    /// Orchestrator stats to track performance
    struct OrchestratorStats has key, store, copy, drop {
        stake_amount: u64,
        successful_tasks: u64,
        total_compute_time: u64
    }

    /// New struct for delegation
    struct DelegatedOrchestrator has key, store {
        owner: address,
        delegated_addresses: vector<address>
    }

    /// Server registry to track authorized servers
    struct ServerRegistry has key {
        authorized_servers: vector<address>
    }

    /// Events
    #[event]
    struct TaskCreatedEvent has drop, store {
        task_id: vector<u8>,
        client: address,              // Server address
        client_identifier: vector<u8>, // Actual client identifier
        payment_amount: u64
    }

    #[event]
    struct TaskCompletedEvent has drop, store {
        task_id: vector<u8>,
        orchestrator: address,
        payment_amount: u64
    }

    #[event]
    struct PaymentEvent has drop, store {
        task_id: vector<u8>,
        orchestrator: address,
        amount: u64
    }

    #[event]
    struct OrchestratorRegisteredEvent has drop, store {
        orchestrator: address,
        stake_amount: u64
    }

    #[event]
    struct DelegateAddedEvent has drop, store {
        owner: address,
        delegate: address
    }

    #[event]
    struct ServerAuthorizedEvent has drop, store {
        server: address
    }

    struct TaskEvent has store, drop {
        task_id: vector<u8>,
        client: address,
        client_identifier: vector<u8>,
        orchestrator: address,
        status: u8,
        timestamp: u64
    }

    struct TaskStore has key {
        tasks: vector<Task>
    }

    /// Initialize the module
    fun init_module(account: &signer) {
        move_to(account, TaskStore {
            tasks: vector[]
        });
        
        move_to(account, ServerRegistry {
            authorized_servers: vector[]
        });
    }

    /// Test initialization
    #[test_only]
    public fun initialize_for_testing(account: &signer) {
        move_to(account, TaskStore {
            tasks: vector[]
        });
        
        move_to(account, ServerRegistry {
            authorized_servers: vector[]
        });
    }

    /// Register a server to handle task creation
    public fun register_server(admin: &signer, server_address: address) acquires ServerRegistry {
        // Only the module account can register servers
        assert!(signer::address_of(admin) == @cactus, EINVALID_CLIENT);
        
        let server_registry = borrow_global_mut<ServerRegistry>(@cactus);
        
        // Check if server is already authorized
        let i = 0;
        let len = vector::length(&server_registry.authorized_servers);
        while (i < len) {
            if (*vector::borrow(&server_registry.authorized_servers, i) == server_address) {
                assert!(false, EALREADY_AUTHORIZED);
            };
            i = i + 1;
        };
        
        vector::push_back(&mut server_registry.authorized_servers, server_address);
        
        // Emit server authorized event
        event::emit(ServerAuthorizedEvent {
            server: server_address
        });
    }

    /// Check if an address is an authorized server
    public fun is_authorized_server(server_address: address): bool acquires ServerRegistry {
        let server_registry = borrow_global<ServerRegistry>(@cactus);
        
        let i = 0;
        let len = vector::length(&server_registry.authorized_servers);
        while (i < len) {
            if (*vector::borrow(&server_registry.authorized_servers, i) == server_address) {
                return true
            };
            i = i + 1;
        };
        
        false
    }

    /// Get task information
    public fun get_task(_task_id: vector<u8>): Task acquires TaskStore {
        let store = borrow_global<TaskStore>(@cactus);
        *vector::borrow(&store.tasks, 0) // For simplicity, just return the first task
    }

    /// Get orchestrator stats
    public fun get_orchestrator_stats(orchestrator: address): OrchestratorStats acquires OrchestratorStats {
        assert!(exists<OrchestratorStats>(orchestrator), ENOT_REGISTERED);
        *borrow_global<OrchestratorStats>(orchestrator)
    }

    // Getter functions for Task fields
    public fun get_task_status(task: &Task): u8 { task.status }
    public fun get_task_payment_amount(task: &Task): u64 { task.payment_amount }
    public fun get_task_orchestrator(task: &Task): address { task.orchestrator }
    public fun get_task_client(task: &Task): address { task.client }
    public fun get_task_client_identifier(task: &Task): vector<u8> { *&task.client_identifier }
    public fun get_task_result_hash(task: &Task): vector<u8> { *&task.result_hash }

    // Getter functions for OrchestratorStats fields
    public fun get_orchestrator_stake_amount(stats: &OrchestratorStats): u64 { stats.stake_amount }
    public fun get_orchestrator_successful_tasks(stats: &OrchestratorStats): u64 { stats.successful_tasks }
    public fun get_orchestrator_total_compute_time(stats: &OrchestratorStats): u64 { stats.total_compute_time }

    /// Check if an address is registered as an orchestrator
    public fun is_registered(orchestrator_address: address): bool {
        exists<OrchestratorStats>(orchestrator_address)
    }

    /// Check if an address is a delegated orchestrator
    public fun is_delegated_orchestrator(delegate_address: address, owner_address: address): bool acquires DelegatedOrchestrator {
        if (!exists<DelegatedOrchestrator>(owner_address)) {
            return false
        };
        
        let delegated = borrow_global<DelegatedOrchestrator>(owner_address);
        let i = 0;
        let len = vector::length(&delegated.delegated_addresses);
        
        while (i < len) {
            if (*vector::borrow(&delegated.delegated_addresses, i) == delegate_address) {
                return true
            };
            i = i + 1;
        };
        
        false
    }

    /// Check if an address is a delegated orchestrator
    public fun is_delegated(owner_address: address, delegate_address: address): bool acquires DelegatedOrchestrator {
        is_delegated_orchestrator(delegate_address, owner_address)
    }

    /// Create a new task and deposit payment - called by the server on behalf of a client
    public fun create_task(
        server: &signer,
        client_identifier: vector<u8>,
        _task_id: vector<u8>,
        orchestrator: address,
        compute_time: u64,
        gpu_resources: u64,
        model_type: vector<u8>,
        payment_amount: u64
    ) acquires OrchestratorStats, TaskStore, ServerRegistry {
        let server_addr = signer::address_of(server);
        
        // Ensure the server is authorized
        assert!(is_authorized_server(server_addr), EUNAUTHORIZED_SERVER);
        
        // Ensure the orchestrator is registered
        assert!(exists<OrchestratorStats>(orchestrator), ENOT_REGISTERED);
        let stats = borrow_global<OrchestratorStats>(orchestrator);
        assert!(stats.stake_amount >= payment_amount, EINSUFFICIENT_BALANCE);

        let task = Task {
            client: server_addr,
            client_identifier,
            orchestrator,
            compute_time,
            gpu_resources,
            model_type,
            payment_amount,
            result_hash: vector[],
            status: STATUS_PENDING
        };

        let store = borrow_global_mut<TaskStore>(@cactus);
        vector::push_back(&mut store.tasks, task);
        
        // Emit task created event
        event::emit(TaskCreatedEvent {
            task_id: _task_id,
            client: server_addr,
            client_identifier,
            payment_amount
        });
    }

    /// Complete a task and release payment with compute time metric
    public fun complete_task(
        server: &signer,
        task_id: u64,
        orchestrator_address: address,
        result_hash: vector<u8>,
        compute_time: u64
    ) acquires TaskStore, OrchestratorStats, ServerRegistry {
        let server_addr = signer::address_of(server);
        
        // Ensure the server is authorized
        assert!(is_authorized_server(server_addr), EUNAUTHORIZED_SERVER);
        
        let store = borrow_global_mut<TaskStore>(@cactus);
        assert!(vector::length(&store.tasks) > 0, ENO_TASKS);
        
        let task = vector::borrow_mut(&mut store.tasks, 0); // For simplicity, just use the first task
        
        // Ensure the server completing the task is the same one that created it
        assert!(server_addr == task.client, EINVALID_SERVER);
        
        // Ensure the orchestrator is the one assigned to the task
        assert!(orchestrator_address == task.orchestrator, EINVALID_ORCHESTRATOR);
        
        // Ensure the task is still pending
        assert!(task.status == STATUS_PENDING, EINVALID_STATUS);

        task.status = STATUS_COMPLETED;
        task.result_hash = result_hash;
        task.compute_time = compute_time; // Update compute time from parameter

        let stats = borrow_global_mut<OrchestratorStats>(orchestrator_address);
        stats.successful_tasks = stats.successful_tasks + 1;
        stats.total_compute_time = stats.total_compute_time + compute_time;

        let payment_event = PaymentEvent {
            task_id: bcs::to_bytes(&task_id),
            orchestrator: orchestrator_address,
            amount: task.payment_amount
        };
        event::emit(payment_event);
        
        // Emit task completed event
        event::emit(TaskCompletedEvent {
            task_id: bcs::to_bytes(&task_id),
            orchestrator: orchestrator_address,
            payment_amount: task.payment_amount
        });
    }

    /// Register as an orchestrator with initial stake
    public fun register_orchestrator(orchestrator: &signer, stake_amount: u64) {
        let orchestrator_addr = signer::address_of(orchestrator);
        assert!(!exists<OrchestratorStats>(orchestrator_addr), EALREADY_REGISTERED);
        
        move_to(orchestrator, OrchestratorStats {
            stake_amount,
            successful_tasks: 0,
            total_compute_time: 0
        });
        
        // Create empty delegation list
        move_to(orchestrator, DelegatedOrchestrator {
            owner: orchestrator_addr,
            delegated_addresses: vector::empty<address>()
        });
        
        // Emit registration event
        event::emit(OrchestratorRegisteredEvent {
            orchestrator: orchestrator_addr,
            stake_amount
        });
    }

    /// Add a delegated orchestrator address
    public fun add_delegated_orchestrator(
        owner: &signer,
        delegate_address: address
    ) acquires DelegatedOrchestrator {
        let owner_addr = signer::address_of(owner);
        
        // Ensure the owner is registered
        assert!(exists<OrchestratorStats>(owner_addr), ENOT_REGISTERED);
        
        // Add the delegate to the list
        let delegated = borrow_global_mut<DelegatedOrchestrator>(owner_addr);
        vector::push_back(&mut delegated.delegated_addresses, delegate_address);
        
        // Emit event
        event::emit(DelegateAddedEvent {
            owner: owner_addr,
            delegate: delegate_address
        });
    }
} 