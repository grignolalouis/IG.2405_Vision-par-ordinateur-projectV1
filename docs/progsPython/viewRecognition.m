%% Visualize the sign recognition in a given database
%% BD            : name (string) of the matfile containing the recognition
%% type          : 'Test' if processing the test database, 'Learn' otherwise
%% resizeFactor  : factor for resizing the bounding boxes. For example 2 if
%%                 the source images have been reduced by a factor 0.5
%%-------------------------------------------------------------------------
function varargout = viewRecognition(BD,type,resizeFactor)

clc
close all
winpos = [1           41         1536        801.6];  % Figure parameters (Position and size)



    
hMainFigure = figure(...       % The main GUI figure
                    'MenuBar','none', ...
                    'Toolbar','none', ...
                    'Color', get(0,'defaultuicontrolbackgroundcolor'), ...
                    'Position',winpos, ...
                    'name','METRO', ...
                    'Color', get(0,'defaultuicontrolbackgroundcolor')...
                    );


hPreviousImageButton  =   uicontrol(...    % button for updating selected plot 
                        'Parent', hMainFigure, ...
                        'Units','normalized',...
                        'HandleVisibility','callback', ...
                        'Position',[0.01 0.07 0.1 0.07],...
                        'String','PREVIOUS IMAGE',...                                         
                        'Visible', 'on', ...
                        'Callback', @hPreviousImageCallback);                
hNextImageButton  =   uicontrol(...    % button for updating selected plot 
                        'Parent', hMainFigure, ...
                        'Units','normalized',...
                        'HandleVisibility','callback', ...
                        'Position',[0.13 0.07 0.1 0.07],...
                        'String','NEXT IMAGE',...                                         
                        'Visible', 'on', ...
                        'Callback', @hNextImageCallback);
                    

hPreviousSymbolButton  =   uicontrol(...    % button for updating selected plot 
                        'Parent', hMainFigure, ...
                        'Units','normalized',...
                        'HandleVisibility','callback', ...
                        'Position',[0.55 0.09 0.1 0.07],...
                        'String','PREVIOUS SYMBOL',...                                         
                        'Visible', 'on', ...
                        'Callback', @hPreviousSymbolCallback);                
hNextSymbolButton  =   uicontrol(...    % button for updating selected plot 
                        'Parent', hMainFigure, ...
                        'Units','normalized',...
                        'HandleVisibility','callback', ...
                        'Position',[0.67 0.09 0.1 0.07],...
                        'String','NEXT SYMBOL',...                                         
                        'Visible', 'on', ...
                        'Callback', @hNextSymbolCallback);    
 
                    
          
load(BD)    ;           % Load the database : BD (name of the variable)


% Define the indices of the images to be processed 

ok = 1;

n = 1:261;
if strcmp(type,'Test')
    numImages  = n(find(mod(n,3)));
elseif strcmp(type,'Learn')
    numImages  = n(find(~mod(n,3)));
else
    ok = 0;
    uiwait(errordlg('Bad identifier (should be ''Learn'' or ''Test'' ','ERRORDLG'));
end

if ok
    ni      = 1;
    nimax   = length(numImages); 
    n       = numImages(ni);
    [L,l,subImages,bbox,hyps,im,nom] = processImage(n);
    viewSymbol(l,L);
end

function  [L,l,subImages,bbox,hyps,im,nom] =  processImage(n)  
    

    %% -------------- READ THE SOURCE IMAGE -----------------------------------

         
    nom         = ['IM (' num2str(n) ')'];
    im          = im2double(imread(['BD' filesep nom '.JPG']));

    [H,W,P]     = size(im);
    

%% -------------- REGION OF INTEREST --------------------------------------
    
    [subImages,bbox,hyps] = loadInformation(BD,n,im);        % Load the bounding boxes and classes
    
   
    if ~isempty(hyps)
        l = 1;  
        L = length(hyps);
    else
        l = 0;
        L = 0;
    end

     
end 

function [subImages,bbox,hyps] = loadInformation(BD,n,im)
    
    % Find the signs recognized in image n
    ind = find(BD(:,1) == n);
    
    if ~isempty(ind)
        bbox = round(resizeFactor * BD(ind,2:5));
        hyps = BD(ind,6);
        subImages = cell(1,length(ind));
        for l = 1:length(ind)
            subImages{l} = im(bbox(l,1):bbox(l,2),bbox(l,3):bbox(l,4),:);
        end    
    else
        subImages = {};
        bbox = [];
        hyps = [];
    end
end
    

function drawRectangle(i1,i2,j1,j2, color)

    hold on;
    i = i1:i2;
    m = length(i);
    plot(i,j1*ones(1,m),color);
    plot(i,j2*ones(1,m),color);


    j = j1:j2;
    m = length(j);
    plot(i1*ones(1,m),j,color);
    plot(i2*ones(1,m),j,color);
end

function viewSymbol(l,L)
    if  l == 0 || L == 0
        subplot(1,2,1); imshow(im);
        title (sprintf('IMAGE %3d : %d SYMBOLES  --  S %2d',n,L,l),'Fontsize',20,'Fontweight','bold');
        
        subplot(1,2,2); imshow([]);
        title (sprintf('LINE = -'),'Fontsize',20,'Fontweight','bold');
        set(gcf,'Position',winpos);
    else
        subplot(1,2,1); imshow(im);
        title (sprintf('IMAGE %3d : %d SYMBOLES  --  S %2d',n,L,l),'Fontsize',20,'Fontweight','bold');
        for k =1:L
            if k == l
                drawRectangle( bbox(l,3),bbox(l,4),bbox(l,1),bbox(l,2),'g') ;
            else
                drawRectangle( bbox(k,3),bbox(k,4),bbox(k,1),bbox(k,2),'r') ;
            end
        end

        
        subplot(1,2,2); imshow(subImages{l});
        title (sprintf('LINE = %2d',hyps(l)),'Fontsize',20,'Fontweight','bold');
        set(gcf,'Position',winpos);
    end
    
end
        
function hPreviousImageCallback(hObject, eventdata)
    
    ni  = max(1,ni-1);
    n   = numImages(ni);
    [L,l,subImages,bbox,hyps,im,nom] = processImage(n);
    viewSymbol(l,L);
end

function hNextImageCallback(hObject, eventdata)
        
    ni = min(nimax,ni+1);
    n   = numImages(ni);
    [L,l,subImages,bbox,hyps,im,nom] = processImage(n);
    viewSymbol(l,L);
    
end

function hPreviousSymbolCallback(hObject, eventdata)
    
    l = max(1,l-1);    
    viewSymbol(l,L) ;
end

function hNextSymbolCallback(hObject, eventdata)
    
   l = min(L,l+1);
    viewSymbol(l,L) ;
end



end