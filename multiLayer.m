% SPI process
classdef multiLayer < nnet.layer.Layer
    
    properties
        % (Optional) Layer properties.
        
        % Layer properties go here.
        muend        
    end

    methods
        function layer = multiLayer(A,NameValueArgs)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            arguments
                A
                NameValueArgs.Name = ''
            end
            % Set layer name.
            layer.Name = NameValueArgs.Name;
            
            % Set layer description.
            layer.Description = "multi a known number";
            
            % Set multiplier.
            layer.muend = A;
            
        end
        
        function Z = predict(layer, X)
            X=X(:); 
            A = layer.muend;
            Z = A*X;
            Z=Z';
            Z=reshape(Z,[1,1,2000]);%2000 is sampling number N


        end

    end
end