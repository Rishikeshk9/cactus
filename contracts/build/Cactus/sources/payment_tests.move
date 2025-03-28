#[test_only]
module cactus::payment_tests {
    #[test_only]
    use aptos_framework::account::create_account_for_test;
    #[test_only]
    use aptos_framework::timestamp::set_time_has_started_for_testing;
    use cactus::payment;

    #[test_only]
    fun setup_test() {
        let framework = create_account_for_test(@0x1);
        let cactus = create_account_for_test(@cactus);
        payment::initialize_for_testing(&cactus);
        set_time_has_started_for_testing(&framework);
    }
    
    #[test]
    fun test_register_server() {
        setup_test();
        let cactus = create_account_for_test(@cactus);
        let server_addr = @0x789;
        
        // Register server
        payment::register_server(&cactus, server_addr);
        
        // Check if server is authorized
        assert!(payment::is_authorized_server(server_addr), 0);
    }
    
    #[test]
    #[expected_failure(abort_code = payment::EINVALID_CLIENT, location = cactus::payment)]
    fun test_register_server_unauthorized() {
        setup_test();
        let unauthorized = create_account_for_test(@0x654);
        let server_addr = @0x789;
        
        // Try to register server from unauthorized account
        payment::register_server(&unauthorized, server_addr);
    }
    
    #[test]
    #[expected_failure(abort_code = payment::EALREADY_AUTHORIZED, location = cactus::payment)]
    fun test_register_server_already_authorized() {
        setup_test();
        let cactus = create_account_for_test(@cactus);
        let server_addr = @0x789;
        
        // Register server first time
        payment::register_server(&cactus, server_addr);
        
        // Try to register again
        payment::register_server(&cactus, server_addr);
    }

    #[test]
    fun test_register_orchestrator() {
        setup_test();
        let orchestrator = create_account_for_test(@0x123);
        
        // Test successful registration
        payment::register_orchestrator(&orchestrator, 1000);
        let stats = payment::get_orchestrator_stats(@0x123);
        assert!(payment::get_orchestrator_stake_amount(&stats) == 1000, 0);
        assert!(payment::get_orchestrator_successful_tasks(&stats) == 0, 1);
        assert!(payment::get_orchestrator_total_compute_time(&stats) == 0, 2);
        assert!(payment::is_registered(@0x123), 3);
    }

    #[test]
    #[expected_failure(abort_code = payment::EALREADY_REGISTERED, location = cactus::payment)]
    fun test_register_orchestrator_already_registered() {
        setup_test();
        let orchestrator = create_account_for_test(@0x123);
        
        // First registration should succeed
        payment::register_orchestrator(&orchestrator, 1000);
        
        // Second registration should fail
        payment::register_orchestrator(&orchestrator, 2000);
    }

    #[test]
    fun test_delegation() {
        setup_test();
        let owner = create_account_for_test(@0x123);
        let _delegate = create_account_for_test(@0x456);

        // Register orchestrator
        payment::register_orchestrator(&owner, 1000);
        
        // Add delegate
        payment::add_delegated_orchestrator(&owner, @0x456);
        
        // Verify delegation
        assert!(payment::is_delegated(@0x123, @0x456), 0);
    }

    #[test]
    #[expected_failure(abort_code = payment::ENOT_REGISTERED, location = cactus::payment)]
    fun test_add_delegate_not_registered() {
        setup_test();
        let owner = create_account_for_test(@0x123);
        
        // Try to add delegate without registering first
        payment::add_delegated_orchestrator(&owner, @0x456);
    }

    #[test]
    fun test_create_task() {
        setup_test();
        let orchestrator = create_account_for_test(@0x123);
        let cactus = create_account_for_test(@cactus);
        let server = create_account_for_test(@0x456);
        
        // Setup
        payment::register_server(&cactus, @0x456);
        payment::register_orchestrator(&orchestrator, 2000);
        
        // Create task
        payment::create_task(
            &server,
            b"client001", // client identifier
            b"task1", // task_id
            @0x123, // orchestrator address
            60, // compute_time
            10, // gpu_resources
            b"stable-diffusion-xl", // model_type
            100 // payment_amount
        );
        
        // Get task and verify details
        let task = payment::get_task(b"task1");
        assert!(payment::get_task_status(&task) == 0, 0); // 0 = created
        assert!(payment::get_task_orchestrator(&task) == @0x123, 0);
        assert!(payment::get_task_client(&task) == @0x456, 0);
        assert!(payment::get_task_payment_amount(&task) == 100, 0);
    }

    #[test]
    #[expected_failure(abort_code = payment::EUNAUTHORIZED_SERVER, location = cactus::payment)]
    fun test_create_task_unauthorized_server() {
        setup_test();
        let orchestrator = create_account_for_test(@0x123);
        let unauthorized_server = create_account_for_test(@0x456);
        
        // Setup - but don't register server
        payment::register_orchestrator(&orchestrator, 2000);
        
        // Try to create task with unauthorized server
        payment::create_task(
            &unauthorized_server,
            b"client001", // client identifier
            b"task1", // task_id
            @0x123, // orchestrator address
            60, // compute_time
            10, // gpu_resources
            b"stable-diffusion-xl", // model_type
            100 // payment_amount
        );
    }

    #[test]
    #[expected_failure(abort_code = payment::ENOT_REGISTERED, location = cactus::payment)]
    fun test_create_task_orchestrator_not_registered() {
        setup_test();
        let cactus = create_account_for_test(@cactus);
        let server = create_account_for_test(@0x456);
        
        // Setup - but don't register orchestrator
        payment::register_server(&cactus, @0x456);
        
        // Try to create task with unregistered orchestrator
        payment::create_task(
            &server,
            b"client001", // client identifier
            b"task1", // task_id
            @0x123, // orchestrator address
            60, // compute_time
            10, // gpu_resources
            b"stable-diffusion-xl", // model_type
            100 // payment_amount
        );
    }

    #[test]
    #[expected_failure(abort_code = payment::EINSUFFICIENT_BALANCE, location = cactus::payment)]
    fun test_create_task_insufficient_stake() {
        setup_test();
        let cactus = create_account_for_test(@cactus);
        let server = create_account_for_test(@0x456);
        
        // Setup with low stake
        payment::register_server(&cactus, @0x456);
        
        // Register orchestrator with very low stake (not enough for task)
        let orchestrator = create_account_for_test(@0x123);
        payment::register_orchestrator(&orchestrator, 5);
        
        // Try to create task with orchestrator that has insufficient stake
        payment::create_task(
            &server,
            b"client001", // client identifier
            b"task1", // task_id
            @0x123, // orchestrator address
            60, // compute_time
            10, // gpu_resources
            b"stable-diffusion-xl", // model_type
            100 // payment_amount
        );
    }

    #[test]
    fun test_complete_task() {
        setup_test();
        let orchestrator = create_account_for_test(@0x123);
        let cactus = create_account_for_test(@cactus);
        let server = create_account_for_test(@0x456);
        
        // Setup
        payment::register_server(&cactus, @0x456);
        payment::register_orchestrator(&orchestrator, 2000);
        
        // Create task
        payment::create_task(
            &server,
            b"client001", // client identifier
            b"task1", // task_id
            @0x123, // orchestrator address
            60, // compute_time
            10, // gpu_resources
            b"stable-diffusion-xl", // model_type
            100 // payment_amount
        );
        
        // Complete task
        payment::complete_task(
            &server,
            1, // task_id (numeric)
            @0x123, // orchestrator address
            b"result", // result hash
            65 // compute time
        );
        
        // Verify task is completed
        let task = payment::get_task(b"task1");
        assert!(payment::get_task_status(&task) == 1, 0); // 1 = completed
        
        // Verify orchestrator stats updated
        let stats = payment::get_orchestrator_stats(@0x123);
        assert!(payment::get_orchestrator_successful_tasks(&stats) == 1, 0);
        assert!(payment::get_orchestrator_total_compute_time(&stats) == 65, 0);
        
        // Verify result hash
        assert!(payment::get_task_result_hash(&task) == b"result", 0);
    }

    #[test]
    #[expected_failure(abort_code = payment::EINVALID_SERVER, location = cactus::payment)]
    fun test_complete_task_wrong_server() {
        setup_test();
        let orchestrator = create_account_for_test(@0x123);
        let cactus = create_account_for_test(@cactus);
        let server = create_account_for_test(@0x456);
        let wrong_server = create_account_for_test(@0x789);
        
        // Setup
        payment::register_server(&cactus, @0x456);
        payment::register_server(&cactus, @0x789);
        payment::register_orchestrator(&orchestrator, 2000);
        
        // Create task with server
        payment::create_task(
            &server,
            b"client001", // client identifier
            b"task1", // task_id
            @0x123, // orchestrator address
            60, // compute_time
            10, // gpu_resources
            b"stable-diffusion-xl", // model_type
            100 // payment_amount
        );
        
        // Try to complete task with wrong server
        payment::complete_task(
            &wrong_server,
            1, // task_id (numeric)
            @0x123, // orchestrator address
            b"result", // result hash
            65 // compute time
        );
    }

    #[test]
    fun test_complete_task_as_delegate() {
        setup_test();
        let owner = create_account_for_test(@0x123);
        let _delegate = create_account_for_test(@0x456);
        let cactus = create_account_for_test(@cactus);
        let server = create_account_for_test(@0x789);
        
        // Setup
        payment::register_server(&cactus, @0x789);
        payment::register_orchestrator(&owner, 2000);
        payment::add_delegated_orchestrator(&owner, @0x456);
        
        // Create task
        payment::create_task(
            &server,
            b"client001", // client identifier
            b"task1", // task_id
            @0x123, // orchestrator address
            60, // compute_time
            10, // gpu_resources
            b"stable-diffusion-xl", // model_type
            100 // payment_amount
        );
        
        // Server completes task for the orchestrator
        payment::complete_task(
            &server,
            1, // task_id (numeric)
            @0x123, // orchestrator address
            b"result", // result hash
            65 // compute time
        );
        
        // Verify task is completed
        let task = payment::get_task(b"task1");
        assert!(payment::get_task_status(&task) == 1, 0); // 1 = completed
        
        // Verify orchestrator stats updated
        let stats = payment::get_orchestrator_stats(@0x123);
        assert!(payment::get_orchestrator_successful_tasks(&stats) == 1, 0);
        assert!(payment::get_orchestrator_total_compute_time(&stats) == 65, 0);
    }

    #[test]
    #[expected_failure(abort_code = payment::EUNAUTHORIZED_SERVER, location = cactus::payment)]
    fun test_complete_task_unauthorized_server() {
        setup_test();
        let orchestrator = create_account_for_test(@0x123);
        let cactus = create_account_for_test(@cactus);
        let server = create_account_for_test(@0x456);
        let unauthorized = create_account_for_test(@0x789);
        
        // Setup
        payment::register_server(&cactus, @0x456);
        payment::register_orchestrator(&orchestrator, 2000);
        
        // Create task
        payment::create_task(
            &server,
            b"client001", // client identifier
            b"task1", // task_id
            @0x123, // orchestrator address
            60, // compute_time
            10, // gpu_resources
            b"stable-diffusion-xl", // model_type
            100 // payment_amount
        );
        
        // Try to complete task with unauthorized server (not registered)
        payment::complete_task(
            &unauthorized,
            1, // task_id (numeric)
            @0x123, // orchestrator address
            b"result", // result hash
            65 // compute time
        );
    }

    #[test]
    #[expected_failure(abort_code = payment::EINVALID_ORCHESTRATOR, location = cactus::payment)]
    fun test_complete_task_invalid_orchestrator() {
        setup_test();
        let orchestrator = create_account_for_test(@0x123);
        let cactus = create_account_for_test(@cactus);
        let server = create_account_for_test(@0x456);
        
        // Setup
        payment::register_server(&cactus, @0x456);
        payment::register_orchestrator(&orchestrator, 2000);
        
        // Create task
        payment::create_task(
            &server,
            b"client001", // client identifier
            b"task1", // task_id
            @0x123, // orchestrator address
            60, // compute_time
            10, // gpu_resources
            b"stable-diffusion-xl", // model_type
            100 // payment_amount
        );
        
        // Try to complete task with wrong orchestrator address
        payment::complete_task(
            &server,
            1, // task_id (numeric)
            @0x789, // wrong orchestrator address
            b"result", // result hash
            65 // compute time
        );
    }

    #[test]
    #[expected_failure(abort_code = payment::ENO_TASKS, location = cactus::payment)]
    fun test_complete_task_no_tasks() {
        setup_test();
        let orchestrator = create_account_for_test(@0x123);
        
        // Setup with no tasks created
        payment::register_orchestrator(&orchestrator, 2000);
        
        let cactus = create_account_for_test(@cactus);
        let server = create_account_for_test(@0x456);
        payment::register_server(&cactus, @0x456);
        
        // Try to complete task that doesn't exist
        payment::complete_task(
            &server,
            1, // non-existent task_id
            @0x123, // orchestrator address
            b"result", // result hash
            65 // compute time
        );
    }

    #[test]
    #[expected_failure(abort_code = payment::EINVALID_STATUS, location = cactus::payment)]
    fun test_complete_task_already_completed() {
        setup_test();
        let orchestrator = create_account_for_test(@0x123);
        let cactus = create_account_for_test(@cactus);
        let server = create_account_for_test(@0x456);
        
        // Setup
        payment::register_server(&cactus, @0x456);
        payment::register_orchestrator(&orchestrator, 2000);
        
        // Create task
        payment::create_task(
            &server,
            b"client001", // client identifier
            b"task1", // task_id
            @0x123, // orchestrator address
            60, // compute_time
            10, // gpu_resources
            b"stable-diffusion-xl", // model_type
            100 // payment_amount
        );
        
        // Complete task first time - should succeed
        payment::complete_task(
            &server,
            1, // task_id (numeric)
            @0x123, // orchestrator address
            b"result", // result hash
            65 // compute time
        );
        
        // Try to complete again - should fail
        payment::complete_task(
            &server,
            1, // task_id (numeric)
            @0x123, // orchestrator address
            b"result2", // different result hash
            70 // different compute time
        );
    }

    #[test]
    #[expected_failure(abort_code = payment::ENOT_REGISTERED, location = cactus::payment)]
    fun test_get_orchestrator_stats_not_registered() {
        setup_test();
        let _owner = create_account_for_test(@0x123);
        
        // Try to get stats for unregistered orchestrator
        payment::get_orchestrator_stats(@0x123);
    }

    #[test]
    fun test_multiple_delegations() {
        setup_test();
        let owner = create_account_for_test(@0x123);
        let _delegate1 = create_account_for_test(@0x456);
        let _delegate2 = create_account_for_test(@0x789);
        let _delegate3 = create_account_for_test(@0x321);
        
        // Register orchestrator
        payment::register_orchestrator(&owner, 1000);
        
        // Add multiple delegates
        payment::add_delegated_orchestrator(&owner, @0x456);
        payment::add_delegated_orchestrator(&owner, @0x789);
        payment::add_delegated_orchestrator(&owner, @0x321);
        
        // Verify delegation
        assert!(payment::is_delegated(@0x123, @0x456), 0);
        assert!(payment::is_delegated(@0x123, @0x789), 0);
        assert!(payment::is_delegated(@0x123, @0x321), 0);
        
        // Verify non-delegation
        assert!(!payment::is_delegated(@0x123, @0x555), 0);
    }

    #[test]
    fun test_get_stats() {
        setup_test();
        let orchestrator = create_account_for_test(@0x123);
        
        // Register orchestrator
        payment::register_orchestrator(&orchestrator, 2000);
        
        // Get stats for newly registered orchestrator
        let stats = payment::get_orchestrator_stats(@0x123);
        assert!(payment::get_orchestrator_stake_amount(&stats) == 2000, 0);
        assert!(payment::get_orchestrator_successful_tasks(&stats) == 0, 0);
        assert!(payment::get_orchestrator_total_compute_time(&stats) == 0, 0);
    }

    #[test]
    fun test_get_task_result_hash() {
        setup_test();
        
        // Create and complete a task
        let orchestrator = create_account_for_test(@0x123);
        let cactus = create_account_for_test(@cactus);
        let server = create_account_for_test(@0x456);
        
        // Setup
        payment::register_server(&cactus, @0x456);
        payment::register_orchestrator(&orchestrator, 2000);
        
        // Create task
        payment::create_task(
            &server,
            b"client001", // client identifier
            b"task1", // task_id
            @0x123, // orchestrator address
            60, // compute_time
            10, // gpu_resources
            b"stable-diffusion-xl", // model_type
            100 // payment_amount
        );
        
        // Complete task with result hash
        payment::complete_task(
            &server,
            1, // task_id (numeric)
            @0x123, // orchestrator address
            b"unique-result-hash", // result hash
            65 // compute time
        );
        
        // Verify result hash can be retrieved
        let task = payment::get_task(b"task1");
        assert!(payment::get_task_result_hash(&task) == b"unique-result-hash", 0);
    }

    #[test]
    fun test_is_delegated_no_delegation() {
        setup_test();
        let owner = create_account_for_test(@0x123);
        
        // Register orchestrator
        payment::register_orchestrator(&owner, 1000);
        
        // Check if address is delegated (should be false)
        assert!(!payment::is_delegated(@0x123, @0x456), 0);
    }
} 