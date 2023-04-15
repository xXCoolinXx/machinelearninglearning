{
  description = "Quantum Chess";
  
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    
    
  };

  outputs = { self, nixpkgs}: 
  let
    # System types to support.
    supportedSystems = [ "x86_64-linux" "x86_64-darwin" ];

    # Helper function to generate an attrset '{ x86_64-linux = f "x86_64-linux"; ... }'.
    forAllSystems = f:
      nixpkgs.lib.genAttrs supportedSystems (system: f system);
    
    # Nixpkgs instantiated for supported system types.
    nixpkgsFor = forAllSystems (system:
      import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        #overlays = [ self.overlay ];
        config.android_sdk.accept_license = true;
      }
    );
  in 
  {
    #packages.x86_64-linux.hello = nixpkgs.legacyPackages.x86_64-linux.hello;

    #packages.x86_64-linux.default = self.packages.x86_64-linux.hello;

    packages = forAllSystems (system:
    let pkgs = nixpkgsFor."${system}";
    in rec {
      ml = pkgs.ml;
    });

    devShell = forAllSystems (system:
    let pkgs = nixpkgsFor."${system}";
    in pkgs.mkShell {
      buildInputs = with pkgs; [ 
        python310
        python310Packages.tensorflow-bin
        python310Packages.keras
        python310Packages.pillow
        #python310Packages.tensorrt 
      ];
    });

    #overlay = final: prev: 
    #  let 
    #    inherit(final)
    #      stdenv lib fetchFromGitHub python311Packages;
    #  in{
    #    python311Packages.tensorflow = prev.python311Packages.tensorflow.overrideAttrs(_:{
    #        src = fetchFromGitHub {
    #          owner = "tensorflow";
    #          repo = "tensorflow";
    #          rev = "refs/tags/v2.12.0";
    #          hash = "sha256-OYh61/83yv+ycivylfdS8yFUIUAk8euAPvmfjPzldGs=";
    #        };
    #        meta.knownVulnerabilities = [];
    #      });
    #  }
    #;
  };
}