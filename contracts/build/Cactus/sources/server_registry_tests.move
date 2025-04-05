#[test_only]
module cactus::server_registry_tests {
    use std::string;
    use std::vector;
    use aptos_framework::account::create_account_for_test;
    use cactus::server_registry;

    #[test_only]
    fun setup_test() {
        let _framework = create_account_for_test(@0x1);
        let cactus = create_account_for_test(@cactus);
        server_registry::initialize(&cactus);
    }
    
    #[test]
    fun test_register_server() {
        setup_test();
        let server = create_account_for_test(@0x123);
        
        // Register server
        server_registry::register_server(
            &server,
            string::utf8(b"127.0.0.1:8080"),
            string::utf8(b"http")
        );
        
        // Check if server is registered
        assert!(server_registry::is_server_registered(@0x123), 0);
        
        // Get server info
        let server_info = server_registry::get_server_by_owner(@0x123);
        assert!(string::utf8(b"127.0.0.1:8080") == server_registry::get_server_ip_port(&server_info), 0);
        assert!(string::utf8(b"http") == server_registry::get_server_protocol(&server_info), 0);
        
        // Check owner
        let owner_address = server_registry::get_server_owner_address(&server_info);
        assert!(owner_address == @0x123, 0);
    }
    
    #[test]
    #[expected_failure(abort_code = server_registry::EALREADY_REGISTERED)]
    fun test_register_server_already_registered() {
        setup_test();
        let server = create_account_for_test(@0x123);
        
        // Register server first time
        server_registry::register_server(
            &server,
            string::utf8(b"127.0.0.1:8080"),
            string::utf8(b"http")
        );
        
        // Try to register again with same owner
        server_registry::register_server(
            &server,
            string::utf8(b"127.0.0.1:8081"),
            string::utf8(b"https")
        );
    }
    
    #[test]
    fun test_update_server_by_owner() {
        setup_test();
        let server = create_account_for_test(@0x123);
        
        // Register server
        server_registry::register_server(
            &server,
            string::utf8(b"127.0.0.1:8080"),
            string::utf8(b"http")
        );
        
        // Update server by owner
        server_registry::update_server(
            &server,
            string::utf8(b"127.0.0.1:8081"),
            string::utf8(b"https"),
            @0x123
        );
        
        // Check updated info
        let server_info = server_registry::get_server_by_owner(@0x123);
        assert!(string::utf8(b"127.0.0.1:8081") == server_registry::get_server_ip_port(&server_info), 0);
        assert!(string::utf8(b"https") == server_registry::get_server_protocol(&server_info), 0);
    }
    
    #[test]
    fun test_update_server_by_admin() {
        setup_test();
        let server = create_account_for_test(@0x123);
        let admin = create_account_for_test(@cactus);
        
        // Register server
        server_registry::register_server(
            &server,
            string::utf8(b"127.0.0.1:8080"),
            string::utf8(b"http")
        );
        
        // Update server by admin
        server_registry::update_server(
            &admin,
            string::utf8(b"127.0.0.1:8081"),
            string::utf8(b"https"),
            @0x123
        );
        
        // Check updated info
        let server_info = server_registry::get_server_by_owner(@0x123);
        assert!(string::utf8(b"127.0.0.1:8081") == server_registry::get_server_ip_port(&server_info), 0);
        assert!(string::utf8(b"https") == server_registry::get_server_protocol(&server_info), 0);
    }
    
    #[test]
    #[expected_failure(abort_code = server_registry::EUNAUTHORIZED)]
    fun test_update_server_by_unauthorized() {
        setup_test();
        let server = create_account_for_test(@0x123);
        let unauthorized = create_account_for_test(@0x456);
        
        // Register server
        server_registry::register_server(
            &server,
            string::utf8(b"127.0.0.1:8080"),
            string::utf8(b"http")
        );
        
        // Try to update server by unauthorized user
        server_registry::update_server(
            &unauthorized,
            string::utf8(b"127.0.0.1:8081"),
            string::utf8(b"https"),
            @0x123
        );
    }
    
    #[test]
    #[expected_failure(abort_code = server_registry::ENOT_REGISTERED)]
    fun test_update_server_not_registered() {
        setup_test();
        let server = create_account_for_test(@0x123);
        
        // Try to update unregistered server
        server_registry::update_server(
            &server,
            string::utf8(b"127.0.0.1:8080"),
            string::utf8(b"http"),
            @0x123
        );
    }
    
    #[test]
    fun test_remove_server_by_owner() {
        setup_test();
        let server = create_account_for_test(@0x123);
        
        // Register server
        server_registry::register_server(
            &server,
            string::utf8(b"127.0.0.1:8080"),
            string::utf8(b"http")
        );
        
        // Remove server by owner
        server_registry::remove_server(&server, @0x123);
        
        // Check if server is removed
        assert!(!server_registry::is_server_registered(@0x123), 0);
    }
    
    #[test]
    fun test_remove_server_by_admin() {
        setup_test();
        let server = create_account_for_test(@0x123);
        let admin = create_account_for_test(@cactus);
        
        // Register server
        server_registry::register_server(
            &server,
            string::utf8(b"127.0.0.1:8080"),
            string::utf8(b"http")
        );
        
        // Remove server by admin
        server_registry::remove_server(&admin, @0x123);
        
        // Check if server is removed
        assert!(!server_registry::is_server_registered(@0x123), 0);
    }
    
    #[test]
    #[expected_failure(abort_code = server_registry::EUNAUTHORIZED)]
    fun test_remove_server_by_unauthorized() {
        setup_test();
        let server = create_account_for_test(@0x123);
        let unauthorized = create_account_for_test(@0x456);
        
        // Register server
        server_registry::register_server(
            &server,
            string::utf8(b"127.0.0.1:8080"),
            string::utf8(b"http")
        );
        
        // Try to remove server by unauthorized user
        server_registry::remove_server(&unauthorized, @0x123);
    }
    
    #[test]
    #[expected_failure(abort_code = server_registry::ENOT_REGISTERED)]
    fun test_remove_server_not_registered() {
        setup_test();
        let server = create_account_for_test(@0x123);
        
        // Try to remove unregistered server
        server_registry::remove_server(&server, @0x123);
    }
    
    #[test]
    fun test_get_all_servers() {
        setup_test();
        let server1 = create_account_for_test(@0x123);
        let server2 = create_account_for_test(@0x456);
        
        // Register servers
        server_registry::register_server(
            &server1,
            string::utf8(b"127.0.0.1:8080"),
            string::utf8(b"http")
        );
        
        server_registry::register_server(
            &server2,
            string::utf8(b"127.0.0.1:8081"),
            string::utf8(b"https")
        );
        
        // Get all servers
        let servers = server_registry::get_all_servers();
        assert!(vector::length(&servers) == 2, 0);
    }
    
    #[test]
    fun test_is_admin() {
        setup_test();
        let admin = create_account_for_test(@cactus);
        let non_admin = create_account_for_test(@0x123);
        
        // Check if admin is admin
        assert!(server_registry::is_admin(@cactus), 0);
        
        // Check if non-admin is admin
        assert!(!server_registry::is_admin(@0x123), 0);
    }
} 